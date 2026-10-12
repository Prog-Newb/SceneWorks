//! Shared Iris task-export weight fixture for this crate's tests.
//!
//! `gen_core::iris` export resolution (`TaskExport::from_dir`, `RestorationResources::from_dir`)
//! reads the `model.safetensors` header for the trained-export `irisTask` stamp and refuses a
//! header it cannot parse, so a staged install's weights must be a real safetensors file. The stub
//! loaders the tests inject never read tensors, so a header-only file is enough.

// Not every target/feature combination compiles every caller.
#![allow(dead_code)]

use gen_core::iris::{IrisTask, ADAPTER_TASK_KEY};

/// A valid, tensor-free `model.safetensors` stamped `irisTask=<task>` in its `__metadata__` (the
/// stamp a trained export of that task carries; a matching stamp resolves, as an unstamped release
/// export does).
pub(crate) fn task_weights(task: IrisTask) -> Vec<u8> {
    let mut header = serde_json::json!({ "__metadata__": { ADAPTER_TASK_KEY: task.name() } })
        .to_string()
        .into_bytes();
    // The safetensors format pads the header with spaces to an 8-byte boundary.
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes
}

/// The released `speridlabs/iris-3b` backbone `config.yaml` (Apache-2.0), so tests that stage an
/// install run the engine's own plan resolution against the real architecture (sc-25689).
pub(crate) const RELEASE_CONFIG_YAML: &str = include_str!("../tests/fixtures/iris/config.yaml");
