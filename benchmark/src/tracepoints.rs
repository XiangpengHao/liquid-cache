use std::sync::OnceLock;

#[usdt::provider]
mod liquid_benchmark {
    fn iteration_start(query_id: u32, iteration: u32) {}
}

static REGISTRATION_SUCCEEDED: OnceLock<bool> = OnceLock::new();

fn ensure_registered() -> bool {
    *REGISTRATION_SUCCEEDED.get_or_init(|| match usdt::register_probes() {
        Ok(()) => true,
        Err(err) => {
            log::debug!("failed to register USDT probes: {err}");
            false
        }
    })
}

pub fn iteration_start(query_id: u32, iteration: u32) {
    if ensure_registered() {
        liquid_benchmark::iteration_start!(|| (query_id, iteration));
    }
}
