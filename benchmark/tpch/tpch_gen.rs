use tpchgen_cli::*;

fn main() {
    let args = Args::parse();
    let mut g = Generator::new(args.scale);
    g.generate();
}