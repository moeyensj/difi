//! difi CLI entry point.
//!
//! Three subcommands: `analyze-observations` (alias `cifi`), `analyze-linkages`
//! (alias `analyze`). The paper's CIFI/DIFI terminology lives in `--help`,
//! log lines, and the `step` field of JSON progress events.

mod analyze_linkages;
mod analyze_observations;
mod common;

use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

use common::RunContext;

#[derive(Parser)]
#[command(
    name = "difi",
    version,
    about = "Did I Find It? — linkage completeness and purity for astronomical surveys",
    long_about = "difi evaluates which objects in a survey are findable (CIFI) and \
                  classifies linkages as pure, contaminated, or mixed (DIFI)."
)]
struct Cli {
    /// Rayon thread count. Defaults to all available CPUs.
    #[arg(long, global = true)]
    threads: Option<usize>,

    /// Emit one JSON progress event per line on stdout. Human-readable text
    /// still goes to stderr.
    #[arg(long, global = true)]
    progress_json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Can I Find It? Compute findability from observations (CIFI).
    #[command(alias = "cifi")]
    AnalyzeObservations(analyze_observations::Args),

    /// Did I Find It? Run CIFI then classify linkages (DIFI). Alias: `analyze`.
    #[command(alias = "analyze")]
    AnalyzeLinkages(analyze_linkages::Args),
}

fn main() -> ExitCode {
    // We want to emit a final `{"event":"error"}` on failure, so we catch the
    // Result rather than using `fn main() -> Result<()>`.
    let cli = Cli::parse();

    if cli.progress_json {
        install_progress_json_panic_hook();
    }

    if let Some(n) = cli.threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            eprintln!("difi: error: failed to configure rayon thread pool: {e}");
            return ExitCode::from(1);
        }
    }

    let ctx = RunContext::new(cli.progress_json, common::argv());

    let result = match cli.command {
        Command::AnalyzeObservations(args) => analyze_observations::run(args, &ctx),
        Command::AnalyzeLinkages(args) => analyze_linkages::run(args, &ctx),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            ctx.emit_error(&e);
            ExitCode::from(exit_code_for(&e))
        }
    }
}

/// Map classified errors to exit codes per the CLI proposal:
///   1 = config/argument error
///   2 = I/O error
///   3 = data error
///
/// Two passes: first look for a `difi::error::Error` anywhere in the chain
/// (library-level classified error), then fall back to a raw `std::io::Error`.
/// The library ordering matters because `parquet`/`arrow` errors can carry an
/// inner `io::Error` we don't want to reclassify as plain I/O.
fn exit_code_for(err: &anyhow::Error) -> u8 {
    for cause in err.chain() {
        if let Some(e) = cause.downcast_ref::<difi::error::Error>() {
            return match e {
                difi::error::Error::InvalidInput(_)
                | difi::error::Error::Parquet(_)
                | difi::error::Error::Arrow(_) => 3,
                difi::error::Error::Io(_) => 2,
            };
        }
    }
    for cause in err.chain() {
        if cause.downcast_ref::<std::io::Error>().is_some() {
            return 2;
        }
    }
    1
}

/// Under `--progress-json`, a library panic would otherwise exit through the
/// default panic hook without emitting any NDJSON event — machine consumers
/// would see the stream stop without a terminal `done`/`error` line.
/// Install a hook that emits an `{"event":"error", ...}` on stdout *before*
/// deferring to the default hook (which still prints to stderr + backtrace).
fn install_progress_json_panic_hook() {
    use std::io::Write;

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "panic with non-string payload".to_string()
        };
        let location = info
            .location()
            .map(|l| format!(" at {}:{}", l.file(), l.line()))
            .unwrap_or_default();
        let event = serde_json::json!({
            "event": "error",
            "message": format!("panicked: {payload}{location}"),
            "ts_unix_s": common::now_unix_s(),
        });
        let _ = writeln!(std::io::stdout(), "{event}");
        default_hook(info);
    }));
}

// Compile-time sanity: prove anyhow::Result is the return type expected by the
// subcommand modules.
const _: fn() -> Result<()> = || Ok(());
