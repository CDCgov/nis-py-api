import argparse
from pathlib import Path

import nisapi

if __name__ == "__main__":
    default_cache = nisapi.root_cache_path()

    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, help=f"Path to cache. Default: {default_cache}")
    sp = p.add_subparsers(dest="subcommand", required=True)
    sp_cache = sp.add_parser("cache", help="Cache all datasets")
    sp_cache.add_argument("--app-token", type=str, help="Socrata developer API token")
    sp_cache.add_argument(
        "--overwrite",
        default="warn",
        choices=["warn"],
        help="Overwrite existing datasets? Only supported option is 'warn', "
        "which prints a notice that a dataset is not overwritten.",
    )
    sp_cache.add_argument(
        "--validate",
        default="warn",
        choices=["warn", "error", "ignore"],
        help="How to handle validation problems?",
    )
    sp_delete = sp.add_parser("delete", help="Delete all cached datasets")
    sp_delete.add_argument(
        "--force",
        action="store_true",
        help="Confirm deletion of all cached datasets",
    )
    args = p.parse_args()

    if args.subcommand == "cache":
        nisapi.cache_all_datasets(
            path=args.path,
            app_token=args.app_token,
            overwrite=args.overwrite,
            validation_mode=args.validate,
        )
    elif args.subcommand == "delete":
        nisapi.delete_cache(path=args.path, confirm=not args.force)
    else:
        print(f"Unknown subcommand: {args.subcommand}")
        p.print_help()
