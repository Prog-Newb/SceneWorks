#[tokio::main]
async fn main() -> Result<(), sceneworks_worker::WorkerError> {
    if std::env::var("SCENEWORKS_GPU_CHECK").is_ok_and(|value| value.trim() == "1") {
        match sceneworks_worker::rocm_preflight() {
            Ok(()) => std::process::exit(0),
            Err(error) => {
                println!("{error}");
                std::process::exit(2);
            }
        }
    }
    sceneworks_worker::run().await
}
