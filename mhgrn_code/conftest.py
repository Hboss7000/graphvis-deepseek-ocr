"""Keep repository-wide pytest discovery out of untouched vendor checkouts."""

collect_ignore_glob = ["GraphVis/*", "GraphVis/**/*", "MHGRN/*", "MHGRN/**/*"]
