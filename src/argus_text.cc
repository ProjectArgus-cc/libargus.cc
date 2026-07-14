/**
 * @file argus_text.cc
 * @brief Model loading, text generation, vocabulary tokenization, KV-cache management,
 *        and speech synthesis engine.
 *
 * Implements the unmanaged execution pathways mapped to the public text handling
 * boundaries of libargus.h, maintaining safe C ABI structures for Project Panama.
 */

#include "libargus.h"
#include "llama.h"

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <mutex>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "argus_internal.h"

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5f * (1.0f - cosf((2.0f * (float)M_PI * i) / (length + offset)));
    }
}

static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2.0f * (float)M_PI * k / N;
    *real = cosf(angle);
    *imag = sinf(angle);
}

static void irfft(int n, const float * inp_cplx, float * out_real, float * scratch_real_inp, float * scratch_imag_inp, float * scratch_real_out, float * scratch_imag_out) {
    int N = n / 2 + 1;

    for (int i = 0; i < N; ++i) {
        scratch_real_inp[i] = inp_cplx[2 * i];
        scratch_imag_inp[i] = inp_cplx[2 * i + 1];
    }

    for (int k = 0; k < n; ++k) {
        scratch_real_out[k] = 0.0f;
        scratch_imag_out[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            scratch_real_out[k] += scratch_real_inp[m] * twiddle_real - scratch_imag_inp[m] * twiddle_imag;
            scratch_imag_out[k] += scratch_real_inp[m] * twiddle_imag + scratch_imag_inp[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = scratch_real_out[i] / N;
    }
}

static void fold(const float * data, int64_t data_size, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, float * output) {
    std::fill(output, output + n_out, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < n_out; ++w_col) {
        int64_t start = w_col * n_hop - n_pad;
        int64_t end   = start + n_win;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < n_out && col_idx < data_size) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }
}

static std::string process_text_for_outetts(const std::string & text) {
    std::string processed;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            processed += std::tolower(static_cast<unsigned char>(c));
        } else {
            processed += ' ';
        }
    }
    std::string trimmed;
    bool in_space = false;
    for (char c : processed) {
        if (c == ' ') {
            if (!in_space) {
                trimmed += ' ';
                in_space = true;
            }
        } else {
            trimmed += c;
            in_space = false;
        }
    }
    if (!trimmed.empty() && trimmed.front() == ' ') trimmed.erase(0, 1);
    if (!trimmed.empty() && trimmed.back() == ' ') trimmed.pop_back();

    std::string result;
    for (char c : trimmed) {
        if (c == ' ') {
            result += "<|text_sep|>";
        } else {
            result += c;
        }
    }
    return result;
}

static int32_t tokenize_to_vector(const struct llama_vocab * vocab, const std::string & text, std::vector<llama_token> & output, bool add_bos) {
    int32_t n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, add_bos, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }
    std::vector<llama_token> tokens(n_tokens);
    int32_t result = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), n_tokens, add_bos, true);
    if (result < 0) {
        return result;
    }
    output.insert(output.end(), tokens.begin(), tokens.begin() + result);
    return result;
}

extern "C" {

// =========================================================================
// Model Lifecycle
// =========================================================================

argus_model_t * argus_model_load(const argus_model_params_t * params) {
    if (!params || !params->model_path) {
        return nullptr;
    }

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params->gpu_layers;
    mparams.use_mlock    = params->use_mlock;

    struct llama_model * model = llama_model_load_from_file(params->model_path, mparams);
    if (!model) {
        return nullptr;
    }

    argus_model_t * argus_model_ptr = new argus_model();
    argus_model_ptr->model = model;
    argus_model_ptr->vocab = llama_model_get_vocab(model);

    return argus_model_ptr;
}

void argus_model_free(argus_model_t * model) {
    if (model) {
        if (model->model) {
            llama_model_free(model->model);
        }
        delete model;
    }
}

// =========================================================================
// Context Lifecycle
// =========================================================================

argus_context_t * argus_context_init(argus_model_t * model, const argus_context_params_t * params) {
    if (!model || !params) {
        return nullptr;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = params->context_length;
    cparams.n_batch         = (params->context_length < 2048) ? params->context_length : 2048;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = 4;
    cparams.n_threads       = params->cpu_threads;
    cparams.n_threads_batch = params->cpu_threads;

    // Multi-Token Prediction (MTP) context type configuration
    cparams.ctx_type = params->enable_draft_mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;

    // KV cache quantization support
    cparams.type_k = (enum ggml_type)params->type_k;
    cparams.type_v = (enum ggml_type)params->type_v;

    // Embeddings support
    cparams.embeddings = params->embeddings;

    struct llama_context * ctx = llama_init_from_model(model->model, cparams);
    if (!ctx) {
        return nullptr;
    }

    struct llama_context * draft_ctx = nullptr;
    if (params->draft_model) {
        struct llama_context_params dparams = llama_context_default_params();
        dparams.n_ctx           = params->context_length;
        dparams.n_batch         = cparams.n_batch;
        dparams.n_ubatch        = cparams.n_ubatch;
        dparams.n_seq_max       = cparams.n_seq_max;
        dparams.n_threads       = params->cpu_threads;
        dparams.n_threads_batch = params->cpu_threads;
        dparams.type_k          = cparams.type_k;
        dparams.type_v          = cparams.type_v;

        draft_ctx = llama_init_from_model(params->draft_model->model, dparams);
    }

    argus_context_t * argus_ctx = new argus_context();
    argus_ctx->ctx              = ctx;
    argus_ctx->draft_ctx        = draft_ctx;
    argus_ctx->model_ref        = model;
    argus_ctx->draft_model_ref  = const_cast<argus_model_t *>(params->draft_model);
    argus_ctx->spec_draft_n_max = params->spec_draft_n_max;
    argus_ctx->enable_draft_mtp = params->enable_draft_mtp;

    return argus_ctx;
}

void argus_context_free(argus_context_t * ctx) {
    if (ctx) {
        if (ctx->ctx) {
            llama_free(ctx->ctx);
        }
        if (ctx->draft_ctx) {
            llama_free(ctx->draft_ctx);
        }
        delete ctx;
    }
}

// =========================================================================
// Tokenizer (Lock-Free, Read-Only Model Vocabulary Operations)
// =========================================================================

int32_t argus_tokenize(const argus_model_t * model, const char * text, int32_t * out_tokens, int32_t max_tokens, bool add_bos) {
    if (!model || !text || !out_tokens || max_tokens <= 0) {
        return -1;
    }

    return llama_tokenize(model->vocab, text, (int32_t)strlen(text), out_tokens, max_tokens, add_bos, true);
}

int32_t argus_token_to_piece(const argus_model_t * model, int32_t token, char * out_buf, int32_t buf_size) {
    if (!model || !out_buf || buf_size <= 0) {
        return -1;
    }

    return llama_token_to_piece(model->vocab, token, out_buf, buf_size, 0, true);
}

int32_t argus_vocab_bos(const argus_model_t * model) {
    return model && model->vocab ? (int32_t)llama_vocab_bos(model->vocab) : -1;
}

int32_t argus_vocab_eos(const argus_model_t * model) {
    return model && model->vocab ? (int32_t)llama_vocab_eos(model->vocab) : -1;
}

int32_t argus_vocab_eot(const argus_model_t * model) {
    return model && model->vocab ? (int32_t)llama_vocab_eot(model->vocab) : -1;
}

int32_t argus_vocab_pad(const argus_model_t * model) {
    return model && model->vocab ? (int32_t)llama_vocab_pad(model->vocab) : -1;
}

int32_t argus_vocab_n_tokens(const argus_model_t * model) {
    return model && model->vocab ? (int32_t)llama_vocab_n_tokens(model->vocab) : -1;
}

bool argus_vocab_is_eog(const argus_model_t * model, int32_t token) {
    return model && model->vocab ? llama_vocab_is_eog(model->vocab, (llama_token)token) : false;
}

int32_t argus_model_meta_val_str(const argus_model_t * model, const char * key, char * buf, int32_t buf_size) {
    if (!model || !model->model || !key || !buf || buf_size <= 0) {
        return -1;
    }
    return llama_model_meta_val_str(model->model, key, buf, (size_t)buf_size);
}

int32_t argus_model_meta_count(const argus_model_t * model) {
    if (!model || !model->model) {
        return -1;
    }
    return llama_model_meta_count(model->model);
}

int32_t argus_model_meta_key_by_index(const argus_model_t * model, int32_t index, char * buf, int32_t buf_size) {
    if (!model || !model->model || !buf || buf_size <= 0 || index < 0) {
        return -1;
    }
    return llama_model_meta_key_by_index(model->model, index, buf, (size_t)buf_size);
}

int32_t argus_model_meta_val_str_by_index(const argus_model_t * model, int32_t index, char * buf, int32_t buf_size) {
    if (!model || !model->model || !buf || buf_size <= 0 || index < 0) {
        return -1;
    }
    return llama_model_meta_val_str_by_index(model->model, index, buf, (size_t)buf_size);
}

int32_t argus_model_n_embd(const argus_model_t * model) {
    return model && model->model ? llama_model_n_embd(model->model) : -1;
}

int32_t argus_model_n_ctx_train(const argus_model_t * model) {
    return model && model->model ? llama_model_n_ctx_train(model->model) : -1;
}

int32_t argus_model_n_layer(const argus_model_t * model) {
    return model && model->model ? llama_model_n_layer(model->model) : -1;
}

int32_t argus_model_n_head(const argus_model_t * model) {
    return model && model->model ? llama_model_n_head(model->model) : -1;
}

int32_t argus_model_n_head_kv(const argus_model_t * model) {
    return model && model->model ? llama_model_n_head_kv(model->model) : -1;
}

uint64_t argus_model_n_params(const argus_model_t * model) {
    return model && model->model ? llama_model_n_params(model->model) : 0;
}

// =========================================================================
// Synchronized Context Operations
// =========================================================================

int32_t argus_decode_batch(argus_context_t * ctx, const argus_token_batch_t * batch_payload) {
    if (!ctx || !batch_payload || !batch_payload->tokens || batch_payload->n_tokens <= 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    struct llama_batch batch = llama_batch_init(batch_payload->n_tokens, 0, 1);
    batch.n_tokens = batch_payload->n_tokens;

    for (int32_t i = 0; i < batch_payload->n_tokens; ++i) {
        batch.token[i]     = batch_payload->tokens[i];
        batch.pos[i]       = batch_payload->start_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = batch_payload->seq_id;
        // Request logits solely on the terminal processing slot of the batch
        batch.logits[i]    = (batch_payload->request_logits && (i == batch_payload->n_tokens - 1)) ? 1 : 0;
    }

    int32_t result = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);

    return result;
}

int32_t argus_get_embeddings(argus_context_t * ctx, int32_t seq_id, float * out_embeddings, int32_t max_floats) {
    if (!ctx || !out_embeddings || max_floats <= 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    float * embd = llama_get_embeddings_seq(ctx->ctx, seq_id);
    if (!embd) {
        return -1;
    }

    int32_t n_embd = llama_model_n_embd(ctx->model_ref->model);
    if (n_embd > max_floats) {
        return -2;
    }

    std::memcpy(out_embeddings, embd, n_embd * sizeof(float));
    return n_embd;
}

int32_t argus_sample_token(argus_context_t * ctx, int32_t seq_id, float temperature, float repeat_penalty) {
    if (!ctx) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler * sampler = llama_sampler_chain_init(sparams);

    // Apply repetition penalty mitigation parameters
    if (repeat_penalty > 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    }

    // Intercept and inject temperature parameters if above absolute floor boundaries
    if (temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    }

    // Default fallback to greedy calculation pass
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // Pull from index -1 targeting the last populated logits matrix
    int32_t token = llama_sampler_sample(sampler, ctx->ctx, -1);
    (void)seq_id; // Parameter retained for forward compatibility transitions

    llama_sampler_free(sampler);
    return token;
}

int32_t argus_sample_token_with_bias(
    argus_context_t          * ctx, 
    int32_t                    seq_id, 
    float                      temperature, 
    float                      repeat_penalty, 
    const argus_logit_bias_t * biases, 
    int32_t                    bias_count
) {
    if (!ctx) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler * sampler = llama_sampler_chain_init(sparams);

    // Apply repetition penalty mitigation parameters
    if (repeat_penalty > 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    }

    // Apply logit bias sampler if biases are provided (zero-copy direct pass)
    if (biases && bias_count > 0 && ctx->model_ref && ctx->model_ref->vocab) {
        int32_t n_vocab = llama_vocab_n_tokens(ctx->model_ref->vocab);
        llama_sampler_chain_add(sampler, llama_sampler_init_logit_bias(n_vocab, bias_count, (const llama_logit_bias *)biases));
    }

    // Intercept and inject temperature parameters if above absolute floor boundaries
    if (temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    }

    // Default fallback to greedy calculation pass
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // Pull from index -1 targeting the last populated logits matrix
    int32_t token = llama_sampler_sample(sampler, ctx->ctx, -1);
    (void)seq_id; // Parameter retained for forward compatibility transitions

    llama_sampler_free(sampler);
    return token;
}

void argus_kv_cache_clear_slot(argus_context_t * ctx, int32_t seq_id, int32_t p0, int32_t p1) {
    if (!ctx) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);

    // Direct binding route targeting unmanaged sequence tracking cells
    llama_memory_seq_rm(llama_get_memory(ctx->ctx), seq_id, p0, p1);
}

// =========================================================================
// TTS: Speech Synthesis Placeholder
// =========================================================================

int32_t argus_synthesize_speech(
    argus_context_t * ctx,
    const argus_model_t * wavtokenizer_model,
    const char * text,
    int32_t voice_seed,
    float * out_pcm,
    int32_t max_samples,
    float * workspace,
    int64_t workspace_size_floats) {

    if (!ctx || !wavtokenizer_model || !text || !out_pcm || max_samples <= 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->mtx);
    (void)voice_seed;

    // Clear KV cache for sequence 0 to prevent position conflicts on retries
    llama_memory_seq_rm(llama_get_memory(ctx->ctx), 0, -1, -1);

    const struct llama_vocab * vocab = ctx->model_ref->vocab;

    // Default speaker setup matching en_male_1
    std::string audio_text = "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>just<|text_sep|>two<|text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>remarkable<|text_sep|>sure<|text_sep|>i<|text_sep|>have<|text_sep|>some<|text_sep|>critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|text_sep|>the<|text_sep|>gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|text_sep|>really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>lovely<|text_sep|>";
    std::string audio_data = "<|audio_start|>\n"
"the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>\n"
"overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>\n"
"package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>\n"
"from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>\n"
"just<|t_0.25|><|code_start|><|1782|><|1670|><|317|><|786|><|1748|><|631|><|599|><|1155|><|1364|><|1524|><|36|><|1591|><|889|><|1535|><|541|><|440|><|1532|><|50|><|870|><|code_end|>\n"
"two<|t_0.24|><|code_start|><|1681|><|1510|><|673|><|799|><|805|><|1342|><|330|><|519|><|62|><|640|><|1138|><|565|><|1552|><|1497|><|1552|><|572|><|1715|><|1732|><|code_end|>\n"
"people<|t_0.39|><|code_start|><|593|><|274|><|136|><|740|><|691|><|633|><|1484|><|1061|><|1138|><|1485|><|344|><|428|><|397|><|1562|><|645|><|917|><|1035|><|1449|><|1669|><|487|><|442|><|1484|><|1329|><|1832|><|1704|><|600|><|761|><|653|><|269|><|code_end|>\n"
"is<|t_0.16|><|code_start|><|566|><|583|><|1755|><|646|><|1337|><|709|><|802|><|1008|><|485|><|1583|><|652|><|10|><|code_end|>\n"
"pretty<|t_0.32|><|code_start|><|1818|><|1747|><|692|><|733|><|1010|><|534|><|406|><|1697|><|1053|><|1521|><|1355|><|1274|><|816|><|1398|><|211|><|1218|><|817|><|1472|><|1703|><|686|><|13|><|822|><|445|><|1068|><|code_end|>\n"
"remarkable<|t_0.68|><|code_start|><|230|><|1048|><|1705|><|355|><|706|><|1149|><|1535|><|1787|><|1356|><|1396|><|835|><|1583|><|486|><|1249|><|286|><|937|><|1076|><|1150|><|614|><|42|><|1058|><|705|><|681|><|798|><|934|><|490|><|514|><|1399|><|572|><|1446|><|1703|><|1346|><|1040|><|1426|><|1304|><|664|><|171|><|1530|><|625|><|64|><|1708|><|1830|><|1030|><|443|><|1509|><|1063|><|1605|><|1785|><|721|><|1440|><|923|><|code_end|>\n"
"sure<|t_0.36|><|code_start|><|792|><|1780|><|923|><|1640|><|265|><|261|><|1525|><|567|><|1491|><|1250|><|1730|><|362|><|919|><|1766|><|543|><|1|><|333|><|113|><|970|><|252|><|1606|><|133|><|302|><|1810|><|1046|><|1190|><|1675|><|code_end|>\n"
"i<|t_0.08|><|code_start|><|123|><|439|><|1074|><|705|><|1799|><|637|><|code_end|>\n"
"have<|t_0.16|><|code_start|><|1509|><|599|><|518|><|1170|><|552|><|1029|><|1267|><|864|><|419|><|143|><|1061|><|0|><|code_end|>\n"
"some<|t_0.16|><|code_start|><|619|><|400|><|1270|><|62|><|1370|><|1832|><|917|><|1661|><|167|><|269|><|1366|><|1508|><|code_end|>\n"
"critiques<|t_0.60|><|code_start|><|559|><|584|><|1163|><|1129|><|1313|><|1728|><|721|><|1146|><|1093|><|577|><|928|><|27|><|630|><|1080|><|1346|><|1337|><|320|><|1382|><|1175|><|1682|><|1556|><|990|><|1683|><|860|><|1721|><|110|><|786|><|376|><|1085|><|756|><|1523|><|234|><|1334|><|1506|><|1578|><|659|><|612|><|1108|><|1466|><|1647|><|308|><|1470|><|746|><|556|><|1061|><|code_end|>\n"
"about<|t_0.29|><|code_start|><|26|><|1649|><|545|><|1367|><|1263|><|1728|><|450|><|859|><|1434|><|497|><|1220|><|1285|><|179|><|755|><|1154|><|779|><|179|><|1229|><|1213|><|922|><|1774|><|1408|><|code_end|>\n"
"some<|t_0.23|><|code_start|><|986|><|28|><|1649|><|778|><|858|><|1519|><|1|><|18|><|26|><|1042|><|1174|><|1309|><|1499|><|1712|><|1692|><|1516|><|1574|><|code_end|>\n"
"of<|t_0.07|><|code_start|><|197|><|716|><|1039|><|1662|><|64|><|code_end|>\n"
"the<|t_0.08|><|code_start|><|1811|><|1568|><|569|><|886|><|1025|><|1374|><|code_end|>\n"
"gameplay<|t_0.48|><|code_start|><|1269|><|1092|><|933|><|1362|><|1762|><|1700|><|1675|><|215|><|781|><|1086|><|461|><|838|><|1022|><|759|><|649|><|1416|><|1004|><|551|><|909|><|787|><|343|><|830|><|1391|><|1040|><|1622|><|1779|><|1360|><|1231|><|1187|><|1317|><|76|><|997|><|989|><|978|><|737|><|189|><|code_end|>\n"
"aspects<|t_0.56|><|code_start|><|1423|><|797|><|1316|><|1222|><|147|><|719|><|1347|><|386|><|1390|><|1558|><|154|><|440|><|634|><|592|><|1097|><|1718|><|712|><|763|><|1118|><|1721|><|1311|><|868|><|580|><|362|><|1435|><|868|><|247|><|221|><|886|><|1145|><|1274|><|1284|><|457|><|1043|><|1459|><|1818|><|62|><|599|><|1035|><|62|><|1649|><|778|><|code_end|>\n"
"but<|t_0.20|><|code_start|><|780|><|1825|><|1681|><|1007|><|861|><|710|><|702|><|939|><|1669|><|1491|><|613|><|1739|><|823|><|1469|><|648|><|code_end|>\n"
"its<|t_0.09|><|code_start|><|92|><|688|><|1623|><|962|><|1670|><|527|><|599|><|code_end|>\n"
"still<|t_0.27|><|code_start|><|636|><|10|><|1217|><|344|><|713|><|957|><|823|><|154|><|1649|><|1286|><|508|><|214|><|1760|><|1250|><|456|><|1352|><|1368|><|921|><|615|><|5|><|code_end|>\n"
"really<|t_0.36|><|code_start|><|55|><|420|><|1008|><|1659|><|27|><|644|><|1266|><|617|><|761|><|1712|><|109|><|1465|><|1587|><|503|><|1541|><|619|><|197|><|1019|><|817|><|269|><|377|><|362|><|1381|><|507|><|1488|><|4|><|1695|><|code_end|>\n"
"enjoyable<|t_0.49|><|code_start|><|678|><|501|><|864|><|319|><|288|><|1472|><|1341|><|686|><|562|><|1463|><|619|><|1563|><|471|><|911|><|730|><|1811|><|1006|><|520|><|861|><|1274|><|125|><|1431|><|638|><|621|><|153|><|876|><|1770|><|437|><|987|><|1653|><|1109|><|898|><|1285|><|80|><|593|><|1709|><|843|><|code_end|>\n"
"and<|t_0.15|><|code_start|><|1285|><|987|><|303|><|1037|><|730|><|1164|><|502|><|120|><|1737|><|1655|><|1318|><|code_end|>\n"
"it<|t_0.09|><|code_start|><|848|><|1366|><|395|><|1601|><|1513|><|593|><|1302|><|code_end|>\n"
"looks<|t_0.27|><|code_start|><|1281|><|1266|><|1755|><|572|><|248|><|1751|><|1257|><|695|><|1380|><|457|><|659|><|585|><|1315|><|1105|><|1776|><|736|><|24|><|736|><|654|><|1027|><|code_end|>\n"
"lovely<|t_0.56|><|code_start|><|634|><|596|><|1766|><|1556|><|1306|><|1285|><|1481|><|1721|><|1123|><|438|><|1246|><|1251|><|795|><|659|><|1381|><|1658|><|217|><|1772|><|562|><|952|><|107|><|1129|><|1112|><|467|><|550|><|1079|><|840|><|1615|><|1469|><|1380|><|168|><|917|><|836|><|1827|><|437|><|583|><|67|><|595|><|1087|><|1646|><|1493|><|1677|><|code_end|>\n";

    std::vector<llama_token> prompt_inp;

    // init prompt with BOS
    tokenize_to_vector(vocab, "<|im_start|>\n", prompt_inp, true);

    // add speaker prompt prefix
    tokenize_to_vector(vocab, audio_text, prompt_inp, false);

    // tokenize input text
    std::string prompt_clean = process_text_for_outetts(text);
    tokenize_to_vector(vocab, prompt_clean, prompt_inp, false);

    tokenize_to_vector(vocab, "<|text_end|>\n", prompt_inp, false);

    // add default speaker codes data
    tokenize_to_vector(vocab, audio_data, prompt_inp, false);

    // run decode batch for initial prompt evaluation
    struct llama_batch batch = llama_batch_init(prompt_inp.size() + 256, 0, 1);
    batch.n_tokens = prompt_inp.size();
    for (size_t i = 0; i < prompt_inp.size(); ++i) {
        batch.token[i]     = prompt_inp[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }
    // only request logits on the last token of the prompt batch
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx->ctx, batch) != 0) {
        llama_batch_free(batch);
        return -1;
    }

    std::vector<llama_token> generated_codes;
    int n_past = batch.n_tokens;
    int n_decode = 0;
    const int n_predict = 4096;

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler * sampler = llama_sampler_chain_init(sparams);
    // Apply repeating penalty and temperature
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.1f));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    while (n_decode <= n_predict) {
        llama_token new_token_id = llama_sampler_sample(sampler, ctx->ctx, -1);
        llama_sampler_accept(sampler, new_token_id);

        generated_codes.push_back(new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token_id;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        n_decode++;
        n_past++;

        if (llama_decode(ctx->ctx, batch) != 0) {
            break;
        }
    }
    llama_sampler_free(sampler);
    llama_batch_free(batch);

    // Extract audio codes (filter range [151672, 155772])
    std::vector<llama_token> vocoder_codes;
    for (auto t : generated_codes) {
        if (t >= 151672 && t <= 155772) {
            vocoder_codes.push_back(t - 151672);
        }
    }

    const int n_codes = vocoder_codes.size();
    if (n_codes == 0) {
        return 0;
    }

    // Initialize vocoder context (WavTokenizer)
    struct llama_context_params vocoder_cparams = llama_context_default_params();
    vocoder_cparams.n_ctx    = n_codes + 16;
    vocoder_cparams.n_batch  = n_codes + 16;
    vocoder_cparams.n_ubatch = 512;
    vocoder_cparams.n_threads = 4;
    vocoder_cparams.embeddings = true;

    struct llama_context * vocoder_ctx = llama_init_from_model(wavtokenizer_model->model, vocoder_cparams);
    if (!vocoder_ctx) {
        return -1;
    }

    struct llama_batch vocoder_batch = llama_batch_init(n_codes, 0, 1);
    vocoder_batch.n_tokens = n_codes;
    for (int i = 0; i < n_codes; ++i) {
        vocoder_batch.token[i]     = vocoder_codes[i];
        vocoder_batch.pos[i]       = i;
        vocoder_batch.n_seq_id[i]  = 1;
        vocoder_batch.seq_id[i][0] = 0;
        vocoder_batch.logits[i]    = false;
    }

    if (llama_decode(vocoder_ctx, vocoder_batch) != 0) {
        llama_batch_free(vocoder_batch);
        llama_free(vocoder_ctx);
        return -1;
    }
    llama_synchronize(vocoder_ctx);

    const int n_embd = llama_model_n_embd_out(wavtokenizer_model->model);
    const float * embd = llama_get_embeddings(vocoder_ctx);
    if (!embd) {
        llama_batch_free(vocoder_batch);
        llama_free(vocoder_ctx);
        return -1;
    }

    // Reconstruct spectral audio floats using the zero-allocation workspace allocator
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop) / 2;
    const int n_out = (n_codes - 1) * n_hop + n_win;

    int n_spec = n_embd * n_codes;

    // Single flat memory buffer allocation (zero allocation churn)
    size_t total_floats = n_spec * 3 + (n_codes * n_fft) * 2 + n_out * 2;
    // Add spaces for irfft scratchpad arrays:
    // scratch_real_inp: 641, scratch_imag_inp: 641, scratch_real_out: 1280, scratch_imag_out: 1280
    size_t scratch_floats = 641 * 2 + 1280 * 2;
    size_t required_floats = total_floats + scratch_floats;

    if (!workspace || workspace_size_floats < (int64_t)required_floats) {
        llama_batch_free(vocoder_batch);
        llama_free(vocoder_ctx);
        return -(int32_t)required_floats; // Signal workspace resize required
    }

    float * E = workspace;
    float * S = E + n_spec;
    float * ST = S + n_spec;
    float * res = ST + n_spec;
    float * hann2 = res + (n_codes * n_fft);
    float * audio = hann2 + (n_codes * n_fft);
    float * env = audio + n_out;

    float * scr_real_inp = env + n_out;
    float * scr_imag_inp = scr_real_inp + 641;
    float * scr_real_out = scr_imag_inp + 641;
    float * scr_imag_out = scr_real_out + 1280;

    float hann[n_fft];
    fill_hann_window(n_fft, true, hann);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k * n_codes + l] = embd[l * n_embd + k];
        }
    }

    for (int k = 0; k < n_embd / 2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k)*n_codes + l];
            float phi = E[(k + n_embd / 2)*n_codes + l];

            mag = expf(mag);
            if (mag > 1e2f) {
                mag = 1e2f;
            }
            S[2 * (k * n_codes + l) + 0] = mag * cosf(phi);
            S[2 * (k * n_codes + l) + 1] = mag * sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd / 2; ++k) {
            ST[l * n_embd + 2 * k + 0] = S[2 * (k * n_codes + l) + 0];
            ST[l * n_embd + 2 * k + 1] = S[2 * (k * n_codes + l) + 1];
        }
    }

    // Run IRFFT on each segment using the pre-allocated scratch workspace
    for (int l = 0; l < n_codes; ++l) {
        irfft(n_fft, ST + l * n_embd, res + l * n_fft, scr_real_inp, scr_imag_inp, scr_real_out, scr_imag_out);
        for (int j = 0; j < n_fft; ++j) {
            res[l * n_fft + j] *= hann[j];
            hann2[l * n_fft + j] = hann[j] * hann[j];
        }
    }

    // Fold overlap-add matrices
    fold(res, n_codes * n_fft, n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_codes * n_fft, n_out, n_win, n_hop, n_pad, env);

    int64_t final_size = n_out - 2 * n_pad;
    int32_t count_to_write = (max_samples < final_size) ? max_samples : (int32_t)final_size;

    for (int32_t i = 0; i < count_to_write; ++i) {
        out_pcm[i] = audio[i] / env[i];
    }

    // Zero out first 0.25 seconds (same as in llama.cpp tools/tts/tts.cpp line 1078)
    int32_t silence_samples = 24000 / 4;
    for (int32_t i = 0; i < silence_samples && i < count_to_write; ++i) {
        out_pcm[i] = 0.0f;
    }

    llama_batch_free(vocoder_batch);
    llama_free(vocoder_ctx);

    return count_to_write;
}

} // extern "C"