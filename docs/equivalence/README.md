# Frozen behavioral reference

`reference_0.3.0a1.pkl` is the **current, versioned reference** used by
`python tools/verify_all.py` for behavioral comparisons (SHA-256: `8e3a5ef53fcb917d7dd50996d37febd7e89d5427d36fb911bf2e80c1152c8207`).
The reference was provided with the original `0.3.0a1` project ZIP;
this cleanup has **not** regenerated or altered it.

To avoid tracking repeated binary data in the public source tree, the
older reference files are stored in a separate, optional archive:
`clustelia_referencias_historicas.zip` (distributed alongside the
repository preparation ZIP, **not** as part of the Python package).
The archived `reference_0.3.0.dev0_local.pkl`,
`reference_0.3.0.dev1.pkl` and `reference_0.3.0.dev2.pkl` are byte-for-byte
identical to the current snapshot. Other historical references are
preserved in that archive as separate artifacts.

**Security:** pickle loading can execute arbitrary code. Never load a
pickle from an untrusted source, including downloaded third-party
benchmarks or GitHub attachments.
