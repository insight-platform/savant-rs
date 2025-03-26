fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Define all possible parameters
    const ALL_PARAMS: &[&str] = &[
        "name",
        "version",
        "license",
        "authors",
        "description",
        "homepage",
        "repository",
        "keywords",
        "rust_version",
    ];

    if args.len() <= 1 {
        // If no arguments provided, print all parameters
        print_filtered_info(ALL_PARAMS);
    } else {
        // Skip the first argument (program name)
        let params: Vec<&str> = args[1..].iter().map(|s| s.as_str()).collect();
        print_filtered_info(&params);
    }
}

fn print_filtered_info(params: &[&str]) {
    for param in params {
        match *param {
            "name" => println!(
                "SAVANT_RS_NAME='{}'",
                escape_for_shell(env!("CARGO_PKG_NAME"))
            ),
            "version" => println!(
                "SAVANT_RS_VERSION='{}'",
                escape_for_shell(env!("CARGO_PKG_VERSION"))
            ),
            "license" => {
                if let Some(license) = option_env!("CARGO_PKG_LICENSE") {
                    println!("SAVANT_RS_LICENSE='{}'", escape_for_shell(license));
                }
            }
            "authors" => {
                if let Some(authors) = option_env!("CARGO_PKG_AUTHORS") {
                    println!("SAVANT_RS_AUTHORS='{}'", escape_for_shell(authors));
                }
            }
            "description" => {
                if let Some(description) = option_env!("CARGO_PKG_DESCRIPTION") {
                    println!("SAVANT_RS_DESCRIPTION='{}'", escape_for_shell(description));
                }
            }
            "homepage" => {
                if let Some(homepage) = option_env!("CARGO_PKG_HOMEPAGE") {
                    println!("SAVANT_RS_HOMEPAGE='{}'", escape_for_shell(homepage));
                }
            }
            "repository" => {
                if let Some(repository) = option_env!("CARGO_PKG_REPOSITORY") {
                    println!("SAVANT_RS_REPOSITORY='{}'", escape_for_shell(repository));
                }
            }
            "keywords" => {
                if let Some(keywords) = option_env!("CARGO_PKG_KEYWORDS") {
                    println!("SAVANT_RS_KEYWORDS='{}'", escape_for_shell(keywords));
                }
            }
            "rust_version" => {
                if let Some(rust_version) = option_env!("CARGO_PKG_RUST_VERSION") {
                    println!(
                        "SAVANT_RS_RUST_VERSION='{}'",
                        escape_for_shell(rust_version)
                    );
                }
            }
            _ => eprintln!("Unknown parameter: {}", param),
        }
    }
}

// Escape special characters for shell compatibility
fn escape_for_shell(s: &str) -> String {
    s.replace("'", "'\\''")
}
