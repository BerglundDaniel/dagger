#pragma once

/*
 * Macro used inside kernels to convert row and col indexes to the correct place in the actual memory
 */
#define ROW_COL_TO_POS(matrixPropStruct, row ,col) ( matrixPropStruct.offset (matrixPropStruct.?? * col) + row; ) //TODO total length of row
#define ROW_TO_POS(matrixPropStruct, row) (matrixPropStruct.offset + row)
