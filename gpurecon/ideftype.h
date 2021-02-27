typedef enum IDEF_ErrorCode {
	IDEF_Success=0L,    /* Operation completed successfully             */
	IDEF_DeviceError = -1L,    /* GPU Device error                          */
	IDEF_GenericError = -2L,    /* Unspecified error                            */
	IDEF_InvalidParam = -3L,    /* Invalid parameter                            */
	IDEF_FileNotFoundError = -4L,    /* File not found                            */
	IDEF_FileSizeError = -5L,    /* File Size Error                            */


	IDEF_NotYetImplemented = -99L,   /* The function is not yet implemented            */
}IDEF_ErrorCode;

