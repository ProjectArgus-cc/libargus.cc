# Release operations

A source tag, a validated candidate, and a published release are separate states.
The source-only v1.7.4 release is not proof of binary publication. This pass does
not publish, move tags, or change repository protection settings.

## Contract

`scripts/release/platforms.json` defines eight native modules. Adding a target
requires changing its validator, public asset contract and required verifier set.
Masks are checked against the C ABI: CPU 1, CUDA 3, ROCm 5, Vulkan 9, Metal 17.

One candidate contains:

- Nine Maven modules, each with runtime, sources and Javadoc JARs, POM and Gradle
  module metadata. Release finalization adds detached signatures and MD5/SHA-1/
  SHA-256/SHA-512 sidecars. Mutable repository-level `maven-metadata.xml` is not
  part of this versioned payload.
- `central-bundle.zip`, containing that exact finalized Maven tree.
- 27 public JAR exports and eight uniquely named native ZIPs. Every native ZIP
  contains the binary embedded in its corresponding classifier JAR.
- `manifest.json`, a complete file inventory with sizes, streaming SHA-256
  hashes, source/version, native build identities and public asset paths. It is
  outside the Central bundle and does not recursively hash itself.
- `checksums.txt`, covering public distributables and the manifest, and a public
  signing certificate retained with the internal candidate.

GitHub receives exactly **37 assets**: 27 JARs, eight native ZIPs, the manifest and
checksums. Generated GitHub source archives are outside this contract. Every
final verifier receipt names the same manifest, core JAR, classifier JAR and
loaded native digest, provider, build/ABI identity and exact feature masks.
GPU classifier loading proves compiled support; it does not prove device inference.

## Validation and construction

`validation.yml` is shared by CI and release. It validates workflows/pins,
builds eight native targets, runs CPU/Metal native and Java integration, exercises
JDK 22/25 shared-arena and public composite leases, requires real CPU media, and
runs ASan/UBSan plus targeted TSan. OpenMP is disabled in sanitizer builds to keep
the instrumented thread runtime self-contained; no race/leak suppressions are used.
Instrumented `argus_test` libraries are never staged.

The CUDA Windows build installs/selects MSVC 14.39 (compiler 19.39), matching
[CUDA 12.4's compiler range](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-microsoft-windows/index.html).
It does not bypass NVCC's host-compiler check. Microsoft lists that legacy
[component](https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-enterprise)
as out of support; upgrading the retained CUDA/MSVC pair is a separate toolchain
maintenance decision. Its hosted installation/compilation still needs CI proof.

Artifact selection uses catalog names and exact immutable artifact IDs from the
current run. Packaging runs in a fresh native-free workspace with
`publishAllPublicationsToCandidateRepository`, which was verified to assemble the
publication without depending on integration tests. Tests remain mandatory in
the preceding native-equipped jobs. This uses Gradle's Maven publication model;
NMCP 1.6.1 remains available for existing local tooling, but is not in the release
publisher's task graph.

Signing secrets arrive only in the finalization step after assembly. PRs produce
explicit unsigned development candidates; publication requires signed release
candidates. Finalization imports the configured key into a temporary GnuPG home,
signs existing files, verifies signatures and seals the candidate. It never
regenerates a JAR. Publication also verifies every detached signature.

Each final classifier runs in fresh JVMs with exactly the designated core and
classifier on the classpath, clean extraction directories and strict SPI-only
verification. CPU classifiers perform real tiny-model decode. A negative check
must exit 3 and report the specific feature mismatch; crashes, missing libraries,
absent receipts and timeouts fail. The aggregate gate requires explicit success
from every prerequisite and exactly eight consistent receipts.

Local entry points (run from the repository root):

```bash
python3 -m pip install PyYAML==6.0.2
python3 scripts/release/pins.py
bash scripts/release/lint_workflows.sh
python3 -m unittest discover -s scripts/release/tests -v
python3 scripts/release/preflight.py v1.7.6
# With downloaded native-<catalog-id> artifact directories:
python3 scripts/release/candidate.py stage native-binaries
./gradlew publishAllPublicationsToCandidateRepository -PskipCMake=true
# An unsigned local rehearsal, using the checkout's full source commit:
python3 scripts/release/candidate.py build build/maven native-binaries candidate \
  --source "$(git rev-parse HEAD)" --development
python3 scripts/release/verify_classifier.py candidate linux-amd64-cpu \
  receipts/linux-amd64-cpu.json --development
```

Preflight requires a clean checkout of the existing tag and exact agreement with
`version.txt`. A manual release dispatch must use that tag as its workflow ref;
it cannot select release bytes from an arbitrary version string. External Action
pins are full commits with checked tag correspondence and metadata paths. Major
Action tags can move: update `actions.lock.json` and the documented YAML comments
together when reviewing Dependabot updates; do not relax reference validation.

## Publication and partial failure

The publisher installs no Java/Gradle build toolchain and calls only transport,
inventory, signature and receipt code. Repository reads are the default;
publication alone receives contents/packages write permissions. Configure Central
credentials and the `release` environment according to repository policy. Do not
assume that naming an environment creates approval/protection rules.

1. Verify the candidate, receipts, signature set, all credentials, and remote tag.
2. Create/reuse a GitHub draft bound to the candidate manifest digest. Upload each
   existing asset and download it to verify its digest. Unexpected, conflicting,
   duplicate or partially uploaded assets stop publication.
3. Upload the existing Central bundle as `USER_MANAGED`. Persist the deployment ID,
   poll validation, request publication, and wait for `PUBLISHED`. `VALIDATED` and
   `PUBLISHING` do not satisfy the gate. The protocol follows the
   [Central Publisher API](https://central.sonatype.org/publish/publish-portal-api/).
4. Retrieve every served Maven payload and signature from Central and compare
   hashes, with a bounded propagation wait. Repository-generated checksum
   sidecars may be normalized by the service; the actual payloads/signatures
   must match. Only then expose the verified GitHub draft.
5. Upload the same finalized Maven files to GitHub Packages as a separately
   reported secondary destination. Existing files are accepted only on digest
   equality. Failure marks that destination incomplete and fails the run; it
   does not erase the already successful primary destinations.

Per-version concurrency never cancels an active publisher. The original candidate
and receipts are retained as workflow artifacts for 90 days, subject to repository
retention limits. Destination state is saved after each transition in both
`publication-state.json` and a machine-readable marker in the GitHub release body.
The state marker is a recovery locator, not a substitute for served-byte checks.

## Recovery

Use **Resume retained release** (`recover-release.yml`) with the existing tag and
original Release run ID. It requires a successful original validation gate for
that exact tag commit, resolves the retained candidate/receipt artifact IDs, and
runs the same transport-only publisher. A failed publication can be resumed;
rebuilding a version is not a recovery strategy.

- An uncertain GitHub asset upload is reconciled by listing and hashing existing
  assets on the next run. Conflicting or incomplete assets require inspection;
  the tool never deletes/replaces them silently.
- Before Central upload, state is durably marked `uploading`. If the response is
  lost before a deployment ID is retained, another upload is refused. Find the
  deployment named `libargus-<manifest SHA-256>` in the Central Portal, then provide
  its ID to the recovery workflow. The status response must match that identity.
  Do not supply an unrelated deployment, drop evidence, or upload rebuilt bytes.
- After a polling/propagation timeout, resume the same deployment and candidate.
- A definitive Central client rejection (such as HTTP 401) records `upload_rejected`
  and its status. Correct the credentials/configuration and resume the same
  candidate. Server failures or conflicts retain the uncertain state for inspection.
- After Central/GitHub success but Packages failure, resume to complete Packages.
- If artifacts have expired, recover a previously archived copy with the identical
  manifest and complete payload; a newly built bundle is not equivalent.
- A source-only release without the candidate marker requires manual reconciliation.
  The publisher intentionally refuses to adopt it or move its tag.

There is no distributed atomic transaction across these services. Report partial
completion explicitly. Repository protection was inspected during this pass:
GitHub reported `main` unprotected and no repository rulesets. Workflow files
alone therefore do not enforce who can create/move release tags; maintainers must
configure that repository policy separately.
