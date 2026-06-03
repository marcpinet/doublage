// Dev check: extract every inline <script> from the web UI files and syntax-check them.
// Usage: node scripts/check_html_js.mjs
import { readFileSync } from "node:fs";

let failed = false;
for (const file of ["dubbing/web/index.html", "dubbing/web/preprocess.html"]) {
  const html = readFileSync(file, "utf-8");
  const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
  let m, i = 0;
  while ((m = re.exec(html)) !== null) {
    i += 1;
    try {
      new Function(m[1]);  // parse-only (never executed)
    } catch (e) {
      failed = true;
      console.error(`${file} inline script #${i}: ${e.message}`);
    }
  }
  console.log(`${file}: ${i} inline script(s) checked`);
}
process.exit(failed ? 1 : 0);
