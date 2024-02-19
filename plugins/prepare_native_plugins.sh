#!/usr/bin/env bash

BUILD_MODE=$1
SAVANT_NAME=savant_core
LIBSAVANT_NAME=lib$SAVANT_NAME.so

case $BUILD_MODE in
    "release")
        ;;
    "debug")
        ;;
    *)
        echo "Unsupported build mode $BUILD_MODE"
        exit 1
        ;;
esac

# stop on error
set -e

# get sha256 for libsavant_core.so
if [[ -z "$CARGO_TARGET_DIR" ]]
then
    CARGO_TARGET_DIR=target
fi

OUTPUT_DIR=$CARGO_TARGET_DIR/$BUILD_MODE

if [[ -f $OUTPUT_DIR/$LIBSAVANT_NAME ]]
then
    echo "Found $LIBSAVANT_NAME in $OUTPUT_DIR"
else
    echo "Error: $LIBSAVANT_NAME not found in $OUTPUT_DIR"
    exit 1
fi

LIBSAVANT_SHA256=$(sha256sum $OUTPUT_DIR/$LIBSAVANT_NAME | awk '{print $1}')
LIBSAVANT_SHA256=${LIBSAVANT_SHA256:0:8}

echo "SHA256 prefix for $LIBSAVANT_NAME: $LIBSAVANT_SHA256"

# check what plugins are available

for PLUGIN in $(ls -1 plugins/native)
do
    if [[ -d plugins/native/$PLUGIN ]]
    then
        echo "Found native PLUGIN $PLUGIN"
        PLUGIN_LIB_NAME=lib$PLUGIN.so
        echo "Checking if $PLUGIN_LIB_NAME exists in $OUTPUT_DIR"
        if [[ -f $OUTPUT_DIR/$PLUGIN_LIB_NAME ]]
        then
            echo "Found $PLUGIN_LIB_NAME in $OUTPUT_DIR"
            EMBEDDED_LIBSAVANT_NAME=$(ldd $OUTPUT_DIR/$PLUGIN_LIB_NAME | grep $SAVANT_NAME | awk '{print $1}')
            echo "Embedded $SAVANT_NAME in $PLUGIN_LIB_NAME: $EMBEDDED_LIBSAVANT_NAME"

            # add sha256 with dash before ".so"
            EMBEDDED_LIBSAVANT_NAME_WITH_SHA256=${EMBEDDED_LIBSAVANT_NAME/.so/-$LIBSAVANT_SHA256.so}
            echo "Embedded $SAVANT_NAME with SHA256 in $PLUGIN_LIB_NAME: $EMBEDDED_LIBSAVANT_NAME_WITH_SHA256"

            # patch elf with new name if not already patched
            if [[ "$EMBEDDED_LIBSAVANT_NAME" == *"-$LIBSAVANT_SHA256"* ]]
            then
                echo "Already patched $PLUGIN_LIB_NAME with $EMBEDDED_LIBSAVANT_NAME_WITH_SHA256"
                continue
            else
                patchelf --replace-needed $EMBEDDED_LIBSAVANT_NAME $EMBEDDED_LIBSAVANT_NAME_WITH_SHA256 $OUTPUT_DIR/$PLUGIN_LIB_NAME
            fi
        else
            echo "Error: $PLUGIN_LIB_NAME not found in $OUTPUT_DIR"
            exit 1
        fi
    else
        echo "Error: $PLUGIN is not a directory"
        exit 1
    fi
done

# pack all plugins in a single tar.gz archive
PLUGINS_ARCHIVE=$OUTPUT_DIR/plugins-$BUILD_MODE-$LIBSAVANT_SHA256-$(uname -p).tar.gz
# remove if exists
rm -f $PLUGINS_ARCHIVE
echo "Packing all plugins in $PLUGINS_ARCHIVE"
for PLUGIN in $(ls -1 plugins/native)
do
    if [[ -d plugins/native/$PLUGIN ]]
    then
        PLUGIN_LIB_NAME=lib$PLUGIN.so
        tar -czf $PLUGINS_ARCHIVE -C $OUTPUT_DIR $PLUGIN_LIB_NAME
    fi
done
