#!/bin/bash

# Batch conversion script for NetCDF to OpenVDB conversion
# Usage: ./batch_convert.sh [input_directory] [output_directory] [variable_list]

# Default settings
INPUT_DIR="${1:-/moonbow/cmdavis4/projects/bl_transport/rams_io/}"
OUTPUT_DIR="${2:-./vdb_output/}"
VARIABLES="${3:-RCP,TRACERP001,THETA}"
CONVERTER="./netcdf_to_vdb"

# Check if converter exists
if [ ! -f "$CONVERTER" ]; then
    echo "Error: Converter executable not found at $CONVERTER"
    echo "Please build the converter first: make"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting batch conversion..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Variables: $VARIABLES"
echo "----------------------------------------"

# Counter for progress tracking
count=0
total=$(find "$INPUT_DIR" -name "*.nc" | wc -l)

# Process all NetCDF files in input directory
for ncfile in "$INPUT_DIR"/*.nc; do
    if [ -f "$ncfile" ]; then
        # Extract filename without path and extension
        basename=$(basename "$ncfile" .nc)
        output_file="$OUTPUT_DIR/${basename}.vdb"
        
        count=$((count + 1))
        echo "[$count/$total] Processing: $basename"
        
        # Run the converter
        "$CONVERTER" "$ncfile" "$output_file" -v "$VARIABLES" --verbose
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully converted to: $output_file"
        else
            echo "  ✗ Failed to convert: $ncfile"
        fi
        echo ""
    fi
done

echo "Batch conversion completed!"
echo "Processed $count files"
echo "Output files are in: $OUTPUT_DIR"

# Optional: Generate summary report
echo "Generating summary..."
echo "File conversion summary:" > "$OUTPUT_DIR/conversion_summary.txt"
echo "Date: $(date)" >> "$OUTPUT_DIR/conversion_summary.txt"
echo "Variables: $VARIABLES" >> "$OUTPUT_DIR/conversion_summary.txt"
echo "Input directory: $INPUT_DIR" >> "$OUTPUT_DIR/conversion_summary.txt"
echo "Files processed: $count" >> "$OUTPUT_DIR/conversion_summary.txt"
echo "" >> "$OUTPUT_DIR/conversion_summary.txt"

# List output files with sizes
echo "Output files:" >> "$OUTPUT_DIR/conversion_summary.txt"
ls -lh "$OUTPUT_DIR"/*.vdb >> "$OUTPUT_DIR/conversion_summary.txt" 2>/dev/null

echo "Summary saved to: $OUTPUT_DIR/conversion_summary.txt"