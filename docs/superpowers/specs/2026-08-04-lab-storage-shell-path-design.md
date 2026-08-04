# Lab Storage Shell Path Migration

## Goal

Update every `Wan2.2_edited` shell-script reference to the retired
`/net/holy-isilon/ifs/rc_labs` storage prefix so it uses `/n/lab_storage`.

## Scope

- Apply an exact prefix substitution in the nine matching `.sh` files.
- Preserve each path suffix, including the `ydu_lab/jaysonzlin/Wan2.2_edited`
  project location and log-file names.
- Do not change non-shell files or unrelated shell-script content.

## Verification

- Confirm no `.sh` files under the workspace contain the old prefix.
- Confirm 27 references to the new prefix exist in the same nine files.
- Run `bash -n` against every edited script.
