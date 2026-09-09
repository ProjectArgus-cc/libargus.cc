"""Fault demonstrations in disposable source copies; never edit the working tree.

Run after a native build has populated pinned dependencies. Compiler failures are
not accepted as evidence: each mutant must build, execute, and fail its assertion.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from catalog import ROOT

def command(args, cwd, log, must_pass=True, timeout=300):
    with log.open('wb') as stream:
        result = subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if must_pass and result.returncode:
        raise RuntimeError('Command failed; inspect ' + str(log))
    return result.returncode

def native(output):
    with tempfile.TemporaryDirectory(prefix='argus-native-faults-') as temporary:
        root = Path(temporary)
        for directory in ('src', 'include', 'cmake', 'tests'): shutil.copytree(ROOT / directory, root / directory)
        for name in ('CMakeLists.txt', 'version.txt'): shutil.copyfile(ROOT / name, root / name)
        command(['cmake', '-B', 'build', '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DARGUS_PORTABLE=ON', '-DARGUS_TESTING=ON',
                 '-DFETCHCONTENT_SOURCE_DIR_LLAMACPP=' + str(ROOT / 'build/_deps/llamacpp-src'),
                 '-DFETCHCONTENT_SOURCE_DIR_WHISPERCPP=' + str(ROOT / 'build/_deps/whispercpp-src')], root, output / 'configure.log')
        def compile(): command(['cmake', '--build', 'build', '--target', 'test_concurrency', '-j4'], root, output / 'compile.log', timeout=1200)
        compile(); command(['build/bin/test_concurrency'], root, output / 'baseline.log')
        cases = [
            ('projector-lock', 'src/argus_execution_lock.h', '    if (!lock.try_lock()) {', '    if (kind == 1) return lock;\n    if (!lock.try_lock()) {', 'gate_cv.wait_for'),
            ('context-lock', 'src/argus_execution_lock.h', '    if (!lock.try_lock()) {', '    if (kind == 2) return lock;\n    if (!lock.try_lock()) {', 'gate_cv.wait_for'),
            ('iterator-duplicate', 'src/argus_multimodal.cc', 'video->test_next++', 'video->test_next', 'values.insert(value).second'),
            ('iterator-drop', 'src/argus_multimodal.cc', '--video->test_remaining;', 'video->test_remaining -= 2;', 'values.size() == 100'),
            ('early-projector-free', 'src/argus_multimodal.cc', '    argus_multimodal_retain(mctx);', '    /* injected missing retain */', 'frees == 0'),
            ('retain-leak', 'src/argus_multimodal.cc', '        argus_test_allocation_checkpoint();',
             '        try { argus_test_allocation_checkpoint(); } catch (...) { owned_model.release(); throw; }', 'model_frees == 1')]
        evidence = []
        for name, filename, before, after, assertion in cases:
            path = root / filename; original = path.read_text()
            if before not in original: raise ValueError('Mutation site changed: ' + name)
            path.write_text(original.replace(before, after, 1))
            try:
                compile(); log = output / (name + '.log')
                status = command(['build/bin/test_concurrency'], root, log, must_pass=False, timeout=30)
                if status == 0 or assertion not in log.read_text(errors='replace'):
                    raise AssertionError('Fault escaped or failed for the wrong reason: ' + name)
                evidence.append({'fault': name, 'assertion': assertion, 'exit': status})
            finally: path.write_text(original)
        return evidence

def java(output):
    with tempfile.TemporaryDirectory(prefix='argus-java-faults-') as temporary:
        root = Path(temporary)
        shutil.copytree(ROOT / 'bindings', root / 'bindings', ignore=shutil.ignore_patterns('build', '.gradle'))
        shutil.copytree(ROOT / 'gradle', root / 'gradle')
        shutil.copytree(ROOT / 'include', root / 'include')
        for name in ('gradlew', 'settings.gradle.kts', 'build.gradle.kts', 'version.txt', 'gradle.properties'):
            if (ROOT / name).exists(): shutil.copy2(ROOT / name, root / name)
        args = [str(root / 'gradlew'), ':libargus-core:test', '-PskipCMake=true', '-PnativeTesting=true',
                '-PnativePath=' + str(ROOT / 'build/lib/libargus_test.so'), '--offline', '--console=plain',
                '--tests', '*NativeSafetyTest.nativeDecodeKeepsBufferAndPublicAbortLeaseAlive']
        # This test's fixture path is rooted at the copied checkout.
        shutil.copytree(ROOT / 'tests/data', root / 'tests/data')
        command(args, root, output / 'java-baseline.log')
        path = root / 'bindings/java/libargus-core/src/main/java/cc/projectargus/libargus/ArgusContext.java'
        before = 'return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, lease.handle());'
        after = 'MemorySegment pointer = lease.handle(); lease.close();\n                return decodeBatch(tokensSeg, nTokens, startPos, seqId, requestLogits, pointer);'
        original = path.read_text()
        if before not in original: raise ValueError('Abort lease mutation site changed')
        path.write_text(original.replace(before, after, 1))
        result = command(args, root, output / 'java-omitted-lease.log', must_pass=False)
        reports = root / 'bindings/java/libargus-core/build/test-results/test'
        assertion = 'Thread never queued on the resource lifecycle lock'
        if result == 0 or not any(assertion in p.read_text() for p in reports.glob('*.xml')):
            raise AssertionError('Omitted lease escaped or failed for the wrong reason')
        return [{'fault': 'omitted-public-abort-lease', 'assertion': assertion, 'exit': result}]

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('kind', choices=['native', 'java']); p.add_argument('output', type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    evidence = native(a.output) if a.kind == 'native' else java(a.output)
    (a.output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
