#!/bin/bash
set -e

# Fetch the submodules (Glottolog, ASJP) at their pinned commits
git submodule update --init

# Compile Holman's asjp62x, shipped with the ASJP dataset: LDND distance matrices between ASJP word lists, written as
# MEGA input, as the ASJP team does for its World Language Tree
ASJP_SOFTWARE_DIR=extern/asjp-software
unzip -o -j extern/asjp/cldf/media/ASJPSoftware003.zip "ASJPSoftware003/asjp62x.f" -d $ASJP_SOFTWARE_DIR
gfortran -O2 -std=legacy -o $ASJP_SOFTWARE_DIR/asjp62x $ASJP_SOFTWARE_DIR/asjp62x.f

# MEGA-CC (command-line MEGA) builds the neighbour-joining trees. Its download needs a licence form, so it is installed
# by hand from https://megasoftware.net (Linux CC)
command -v megacc >/dev/null || echo "MEGA-CC (megacc) not found: install it from https://megasoftware.net"
echo "Done"