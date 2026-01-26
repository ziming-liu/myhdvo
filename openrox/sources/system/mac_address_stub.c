#include <generated/config.h>

// Stub implementation for ARM platform
int rox_mac_address_read_decimal(unsigned char *mac_address)
{
    // Return a dummy MAC address for ARM platform
    if (mac_address != NULL)
    {
        mac_address[0] = 0x00;
        mac_address[1] = 0x00;
        mac_address[2] = 0x00;
        mac_address[3] = 0x00;
        mac_address[4] = 0x00;
        mac_address[5] = 0x00;
    }
    return 0;
}
