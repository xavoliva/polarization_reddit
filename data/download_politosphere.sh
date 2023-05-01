YEAR=$1
METADATA=$2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $YEAR -gt 2000 ];
then
    echo $YEAR

    echo "Download comments"

    mkdir -p $SCRIPT_DIR/comments/
    for month in {01..12}
    do
        curl https://zenodo.org/record/5851729/files/comments_$YEAR-$month.bz2?download=1 -o $SCRIPT_DIR/comments/comments_$YEAR-$month.bz2
    done

    echo "Download networks"
    mkdir -p $SCRIPT_DIR/networks

    curl https://zenodo.org/record/5851729/files/networks_$YEAR.csv?download=1 -o $SCRIPT_DIR/networks/networks_$YEAR.csv
fi

if [ $METADATA -eq "1" ];
then
    echo "Download metadata"
    mkdir -p $SCRIPT_DIR/metadata
    curl https://zenodo.org/record/5851729/files/subreddits_metadata.json?download=1 -o $SCRIPT_DIR/metadata/subreddits_metadata.json
    curl https://zenodo.org/record/5851729/files/users_metadata.json?download=1 -o $SCRIPT_DIR/metadata/users_metadata.json
fi
