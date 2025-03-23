use url::Url;

/// Sanitize an object store URL for use as a directory name.
pub fn sanitize_object_store_url_for_dirname(url: &Url) -> String {
    let mut parts = vec![url.scheme()];

    if let Some(host) = url.host_str() {
        parts.push(host);
    }

    let dirname = parts.join("_");

    dirname.replace(['/', ':', '?', '&', '=', '\\'], "_")
}

/// Sanitize a path for use as a directory name.
pub fn sanitize_path_for_dirname(path: &str) -> String {
    path.replace(['/', ':', '?', '&', '=', '\\'], "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use url::Url;

    #[test]
    fn test_can_create_directories_with_sanitized_names() {
        // Create a temporary directory for testing
        let temp_dir = TempDir::new().expect("Failed to create temp directory");

        // Array of problematic URLs to test
        let test_urls = [
            "http://example.com/path/to/resource",
            "https://example.com?param1=value1&param2=value2",
            "s3://bucket-name/object/key",
            "https://user:password@example.com:8080/path?query=value#fragment",
            "file:///C:/Windows/System32/",
            "https://example.com/path/with/special?chars=%20%26%3F",
            "http://192.168.1.1:8080/admin?debug=true",
            "ftp://files.example.com/pub/file.txt",
            // Unicode characters in URL
            "https://例子.测试",
            // Very long URL
            &format!("https://example.com/{}", "a".repeat(200)),
        ];

        // Test each URL
        for url_str in test_urls {
            let url = Url::parse(url_str).expect("Failed to parse URL");
            let dirname = sanitize_object_store_url_for_dirname(&url);

            // Create a directory using the sanitized name
            let dir_path = temp_dir.path().join(dirname);
            fs::create_dir(&dir_path).expect("Failed to create directory");

            // Verify the directory exists
            assert!(dir_path.exists());
            assert!(dir_path.is_dir());

            // Clean up
            fs::remove_dir(&dir_path).expect("Failed to remove test directory");
        }
    }
}
