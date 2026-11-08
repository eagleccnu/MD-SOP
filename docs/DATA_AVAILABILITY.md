# Data availability at release preparation

This is a code, frozen-model and evaluation-summary package, not a complete training-data publication. No repository/DOI or public dataset download endpoint is invented.

Included: record identifiers and frozen source/split/group roles, sequence hashes/lengths, dataset counts, per-protein evaluation metrics, paired statistics, training histories and figures. Primary public tables remove workstation path columns; their numerical values and identities are verified against accepted source tables. Original source file digests are recorded separately.

Not included: MD trajectories, full per-residue MD/S-OPPE/experimental label arrays, feature matrices/cache, raw PDB/NMR downloads, ESM-C pretrained weights, HH-suite search databases, search intermediates, optimizer/last checkpoints, older model weights, private operational records or credential files.

Data redistribution scope, dataset versioning, author contact and durable hosting remain owner decisions before public release. Reference sequences/labels and third-party data must be reviewed under their respective source terms. Users cannot reproduce the full training simply from metadata hashes and score tables. The feature-input inference wrapper is independently usable when compatible precomputed features are supplied.

Do not fill missing targets with zero or select model-specific masks. Frozen target definitions, valid-position counts and protected-reference policy are described in the model card and metadata. Experimental10 is historically reused and not a newly blind external validation set.
