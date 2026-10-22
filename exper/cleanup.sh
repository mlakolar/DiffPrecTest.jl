#!/bin/bash

## removes the GNU parallel job logs and editor backups written next to the
## scripts

cd "$(dirname "$0")"
rm -f *~ sim*/*~ sim*/__*.runtask.log
