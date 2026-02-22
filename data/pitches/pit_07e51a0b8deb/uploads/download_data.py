from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lib import (
    NOAA_WEEKLY_HDD_ARCHIVE_URL,
    Paths,
    download_noaa_weekly_hdd_csv,
    download_static_files,
    ensure_dirs,
)


def main() -> None:
    paths = Paths.from_script(Path(__file__))
    ensure_dirs(paths)

    outputs = download_static_files(paths)

    noaa_out = paths.raw_dir / "noaa_weekly_hdd_us.csv"
    noaa_df = download_noaa_weekly_hdd_csv(noaa_out, start_year=2010)
    outputs["noaa_weekly_hdd_us"] = str(noaa_out)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": {
            "eia_wngsr_latest": "https://ir.eia.gov/ngs/wngsr.csv",
            "eia_ngshistory": "https://ir.eia.gov/ngs/ngshistory.xls",
            "noaa_weekly_hdd_archive": NOAA_WEEKLY_HDD_ARCHIVE_URL,
            "eqt_prices_stooq": "https://stooq.com/q/d/l/?s=eqt.us&i=d",
        },
        "files": outputs,
        "rows": {"noaa_weekly_hdd_us": int(noaa_df.shape[0])},
    }
    (paths.raw_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Download complete.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

