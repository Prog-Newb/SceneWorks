//! `estimate_depth` (Iris-3B monocular depth, sc-25682): a real rmcp client drives the typed
//! `POST /api/v1/image/depth/jobs` submit and fetches the finished depth asset — its raw `.npy` AND
//! its separate preview — through `get_job_result`.

mod common;

use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use rmcp::model::CallToolRequestParams;
use serde_json::{json, Value};

use common::{call_args, connect_mcp, error_text, fast_job_wait, result_json, spawn, TestClient};

type Submitted = Arc<Mutex<Vec<Value>>>;

fn router(submitted: Submitted) -> Router {
    Router::new()
        .route(
            "/api/v1/image/depth/jobs",
            post(|State(submitted): State<Submitted>, Json(body): Json<Value>| async move {
                submitted.lock().unwrap().push(body);
                (
                    StatusCode::CREATED,
                    Json(json!({
                        "id": "job_d1", "type": "image_depth", "status": "queued", "projectId": "p1",
                        "progress": 0.0, "stage": "queued", "message": "", "error": null, "result": {}
                    })),
                )
            }),
        )
        .route(
            "/api/v1/jobs/:job_id",
            get(|Path(_job): Path<String>| async move {
                Json(json!({
                    "id": "job_d1", "type": "image_depth", "status": "completed", "projectId": "p1",
                    "progress": 1.0, "stage": "completed", "message": "", "error": null,
                    "result": { "assets": [{
                        "id": "asset_depth", "type": "depth",
                        "file": { "path": "assets/depth/g/map.npy", "mimeType": "application/x-npy" },
                        "preview": { "path": "assets/depth/g/map.preview.png", "mimeType": "image/png" }
                    }] }
                }))
            }),
        )
        .route(
            "/api/v1/files/ticket",
            post(|| async { Json(json!({ "ticket": "tkt0123456789abcdef", "expiresInSeconds": 600 })) }),
        )
        .with_state(submitted)
}

#[tokio::test]
async fn estimate_depth_submits_the_typed_request_and_links_the_raw_map_and_preview() {
    let submitted: Submitted = Arc::default();
    let api_base = spawn(router(submitted.clone())).await;
    let (_mcp_base, client) = connect_mcp(api_base, None, fast_job_wait(), TestClient).await;

    let result = client
        .call_tool(
            CallToolRequestParams::new("estimate_depth").with_arguments(call_args(
                json!({ "projectId": "p1", "sourceAssetId": "img_1" }),
            )),
        )
        .await
        .expect("estimate_depth succeeds");
    assert_ne!(result.is_error, Some(true), "{result:?}");
    assert_eq!(result_json(&result)["jobId"], "job_d1");
    client
        .call_tool(
            CallToolRequestParams::new("estimate_depth").with_arguments(call_args(json!({
                "projectId": "p1", "sourceAssetId": "img_1", "resolutionMode": "native"
            }))),
        )
        .await
        .expect("native succeeds");
    let bodies = submitted.lock().unwrap().clone();
    assert_eq!(
        bodies[0]["resolution"],
        json!({ "mode": "capped", "maxSide": 1024 })
    );
    assert_eq!(bodies[1]["resolution"], json!({ "mode": "native" }));

    // An unknown mode is refused before anything is submitted.
    let refused = client
        .call_tool(
            CallToolRequestParams::new("estimate_depth").with_arguments(call_args(json!({
                "projectId": "p1", "sourceAssetId": "img_1", "resolutionMode": "metric"
            }))),
        )
        .await;
    match refused {
        Err(_) => {}
        Ok(result) => assert!(error_text(&result).contains("resolutionMode"), "{result:?}"),
    }
    assert_eq!(submitted.lock().unwrap().len(), 2);

    let links = client
        .call_tool(
            CallToolRequestParams::new("get_job_result")
                .with_arguments(call_args(json!({ "jobId": "job_d1" }))),
        )
        .await
        .expect("get_job_result succeeds");
    let uris: Vec<String> = links
        .content
        .iter()
        .filter_map(|block| block.as_resource_link())
        .map(|link| link.uri.clone())
        .collect();
    assert!(uris.iter().any(|uri| uri.contains("map.npy")), "{uris:?}");
    assert!(
        uris.iter().any(|uri| uri.contains("map.preview.png")),
        "{uris:?}"
    );
    let summary = result_json(&links);
    assert_eq!(summary["assets"][0]["mimeType"], "application/x-npy");
    assert!(summary["assets"][0]["previewUrl"]
        .as_str()
        .is_some_and(|url| url.contains("map.preview.png")));
}
