import argparse

import nisapi

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="subcommand")
    sp.add_parser("cache", help="Cache all datasets")
    sp.add_parser("delete", help="Delete all cached datasets")
    args = p.parse_args()

    if args.subcommand == "cache_all":
        nisapi.cache_all_datasets()
    elif args.subcommand == "delete":
        nisapi.delete_cache()
    else:
        p.print_help()
