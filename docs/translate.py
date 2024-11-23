"""Use Google Translate to translate .po files in a directory.

Requirements:
    Requires the googletrans library and the polib library.
    To install them, run:
        pip install googletrans polib

Usage:
    python translate.py <language> <source> --destination <destination>

    <language> is the two-letter code for the destination language.
    <source> is the path to the directory containing the .po files.
    <destination> is the path to the directory to save the translated .po
    files.

How to build the documentation in another language
1. From the docs folder, run `make clean html`
2. Run `make gettext`
    ----> generates .pot files under `build/gettext/`
3. Run `sphinx-intl update -p build/gettext -l de`
     ----> generate .po files under `locales/de/LC_MESSAGES`
4. `mv locales source`
    ---> From the docs, I am not sure why `locales` is not created under
    `source` by default, but otherwise it won't pick up the translations
    in the `.po` files.
5. Add translations to messages in .po files using this script
    python translate.py <language-code> <po-files-directory>
6. Build the html files form the translations
    `make -e SPHINXOPTS="-D language='de'" html`

"""

import argparse
from pathlib import Path

import polib  # type: ignore
from googletrans import Translator


def main(args):
    """Translate .po files in source directory to destination language."""
    # parse args
    language = args.language
    source = args.source
    destination = args.destination

    # set destination to overwrite source files if not provided
    if destination is None:
        destination = source

    # create translator object
    translator = Translator(
        service_urls=[
            "translate.google.com",
            f"translate.google.{language}",
        ]
    )

    # get list of .po files in source directory using pathlib
    list_input_files = [x for x in Path(source).iterdir() if x.suffix == ".po"]
    list_input_files.sort()

    for po_file in list_input_files[1:]:  # skip the first file
        print(f"Translating file: {po_file.name}")
        # create po object from input file
        po = polib.pofile(po_file)

        # create destination po object and copy metadata from input
        po_out = polib.POFile()
        po_out.metadata = po.metadata

        # loop thru untranslated entries in input
        for e, entry in enumerate(po.untranslated_entries()):
            # translate if no errors
            try:
                result = translator.translate(entry.msgid, dest=language).text
            except Exception as e:
                print(
                    f"ERROR {e}: leaving the following "
                    f"translation empty '{entry.msgid}'"
                )
                result = ""

            # create new entry
            new_entry = polib.POEntry(
                msgid=entry.msgid,  # original text
                msgstr=result,  # translated text
                occurrences=entry.occurrences,
            )

            # append new entry to destination po object
            po_out.append(new_entry)

        # save output file
        out_file = Path(destination) / po_file.name
        print(f"Saving output file: {str(out_file)}")
        po_out.save(out_file)


if __name__ == "__main__":
    # make an argument parser
    parser = argparse.ArgumentParser()

    # add language code as a positional argument
    parser.add_argument(
        "language",
        type=str,
        help=(
            "The two letters of the destination language. "
            "E.g., `de` for German, or `es` for Spanish."
        ),
    )
    # add source directory as an argument
    parser.add_argument(
        "source",
        type=str,
        help=(
            "The path to the source directory containing the .po files. "
            "Usually locales/<language>/LC_MESSAGES under source."
        ),
    )
    # add destination directory as an argument
    parser.add_argument(
        "--destination",
        type=str,
        default=None,
        help=(
            "The path to the destination directory to save the translated"
            ".po files."  # ---> by default, same as source?
        ),
    )

    # parse the arguments
    args = parser.parse_args()

    # call the main function
    main(args)
