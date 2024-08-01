import xml.etree.ElementTree as ET
import re
import sys

# 内置的XML数据
XML_DATA = """<?xml version="1.0" encoding="UTF-8"?>
<database xmlns="http://nouveau.freedesktop.org/"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://nouveau.freedesktop.org/ rules-ng.xsd">

<copyright year="2024">

<author name="Tomeu Vizoso" email="tomeu@tomeuvizoso.net"><nick name="tomeu"/>
Initial Author.
</author>

<license>
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice (including the
next paragraph) shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE COPYRIGHT OWNER(S) AND/OR ITS SUPPLIERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
</license>

</copyright>

<enum name="target">
   <value name="PC" value="0x100"/>
   <value name="CNA" value="0x200"/>
   <value name="CORE" value="0x800"/>
   <value name="DPU" value="0x1000"/>
   <value name="DPU_RDMA" value="0x2000"/>
   <value name="PPU" value="0x4000"/>
   <value name="PPU_RDMA" value="0x8000"/>
   <value name="DDMA" value="0x10000"/>
   <value name="SDMA" value="0x20000"/>
   <value name="GLOBAL" value="0x40000"/>
</enum>

<domain name="PC" width="32">
   <reg32 offset="0x0008" name="OPERATION_ENABLE">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="OP_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x0010" name="BASE_ADDRESS">
      <bitfield name="PC_SOURCE_ADDR" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="1" high="3" type="uint"/>
      <bitfield name="PC_SEL" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x0014" name="REGISTER_AMOUNTS">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="PC_DATA_AMOUNT" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x0020" name="INTERRUPT_MASK">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="INT_MASK" low="0" high="16" type="uint"/>
   </reg32>
   <reg32 offset="0x0024" name="INTERRUPT_CLEAR">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="INT_CLR" low="0" high="16" type="uint"/>
   </reg32>
   <reg32 offset="0x0028" name="INTERRUPT_STATUS">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="INT_ST" low="0" high="16" type="uint"/>
   </reg32>
   <reg32 offset="0x002C" name="INTERRUPT_RAW_STATUS">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="INT_RAW_ST" low="0" high="16" type="uint"/>
   </reg32>
   <reg32 offset="0x0030" name="TASK_CON">
      <bitfield name="RESERVED_0" low="14" high="31" type="uint"/>
      <bitfield name="TASK_COUNT_CLEAR" pos="13" type="uint"/>
      <bitfield name="TASK_PP_EN" pos="12" type="uint"/>
      <bitfield name="TASK_NUMBER" low="0" high="11" type="uint"/>
   </reg32>
   <reg32 offset="0x0034" name="TASK_DMA_BASE_ADDR">
      <bitfield name="DMA_BASE_ADDR" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x003C" name="TASK_STATUS">
      <bitfield name="RESERVED_0" low="28" high="31" type="uint"/>
      <bitfield name="TASK_STATUS" low="0" high="27" type="uint"/>
   </reg32>
</domain>
<domain name="CNA" width="32">
   <reg32 offset="0x1000" name="S_STATUS">
      <bitfield name="RESERVED_0" low="18" high="31" type="uint"/>
      <bitfield name="STATUS_1" low="16" high="17" type="uint"/>
      <bitfield name="RESERVED_1" low="2" high="15" type="uint"/>
      <bitfield name="STATUS_0" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x1004" name="S_POINTER">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="EXECUTER" pos="16" type="uint"/>
      <bitfield name="RESERVED_1" low="6" high="15" type="uint"/>
      <bitfield name="EXECUTER_PP_CLEAR" pos="5" type="uint"/>
      <bitfield name="POINTER_PP_CLEAR" pos="4" type="uint"/>
      <bitfield name="POINTER_PP_MODE" pos="3" type="uint"/>
      <bitfield name="EXECUTER_PP_EN" pos="2" type="uint"/>
      <bitfield name="POINTER_PP_EN" pos="1" type="uint"/>
      <bitfield name="POINTER" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x1008" name="OPERATION_ENABLE">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="OP_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x100C" name="CONV_CON1">
      <bitfield name="RESERVED_0" pos="31" type="uint"/>
      <bitfield name="NONALIGN_DMA" pos="30" type="uint"/>
      <bitfield name="GROUP_LINE_OFF" pos="29" type="uint"/>
      <bitfield name="RESERVED_1" low="17" high="28" type="uint"/>
      <bitfield name="DECONV" pos="16" type="uint"/>
      <bitfield name="ARGB_IN" low="12" high="15" type="uint"/>
      <bitfield name="RESERVED_2" low="10" high="11" type="uint"/>
      <bitfield name="PROC_PRECISION" low="7" high="9" type="uint"/>
      <bitfield name="IN_PRECISION" low="4" high="6" type="uint"/>
      <bitfield name="CONV_MODE" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x1010" name="CONV_CON2">
      <bitfield name="RESERVED_0" low="24" high="31" type="uint"/>
      <bitfield name="KERNEL_GROUP" low="16" high="23" type="uint"/>
      <bitfield name="RESERVED_1" low="14" high="15" type="uint"/>
      <bitfield name="FEATURE_GRAINS" low="4" high="13" type="uint"/>
      <bitfield name="RESERVED_2" pos="3" type="uint"/>
      <bitfield name="CSC_WO_EN" pos="2" type="uint"/>
      <bitfield name="CSC_DO_EN" pos="1" type="uint"/>
      <bitfield name="CMD_FIFO_SRST" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x1014" name="CONV_CON3">
      <bitfield name="RESERVED_0" pos="31" type="uint"/>
      <bitfield name="NN_MODE" low="28" high="30" type="uint"/>
      <bitfield name="RESERVED_1" low="26" high="27" type="uint"/>
      <bitfield name="ATROUS_Y_DILATION" low="21" high="25" type="uint"/>
      <bitfield name="ATROUS_X_DILATION" low="16" high="20" type="uint"/>
      <bitfield name="RESERVED_2" low="14" high="15" type="uint"/>
      <bitfield name="DECONV_Y_STRIDE" low="11" high="13" type="uint"/>
      <bitfield name="DECONV_X_STRIDE" low="8" high="10" type="uint"/>
      <bitfield name="RESERVED_3" low="6" high="7" type="uint"/>
      <bitfield name="CONV_Y_STRIDE" low="3" high="5" type="uint"/>
      <bitfield name="CONV_X_STRIDE" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x1020" name="DATA_SIZE0">
      <bitfield name="RESERVED_0" low="27" high="31" type="uint"/>
      <bitfield name="DATAIN_WIDTH" low="16" high="26" type="uint"/>
      <bitfield name="RESERVED_1" low="11" high="15" type="uint"/>
      <bitfield name="DATAIN_HEIGHT" low="0" high="10" type="uint"/>
   </reg32>
   <reg32 offset="0x1024" name="DATA_SIZE1">
      <bitfield name="RESERVED_0" low="30" high="31" type="uint"/>
      <bitfield name="DATAIN_CHANNEL_REAL" low="16" high="29" type="uint"/>
      <bitfield name="DATAIN_CHANNEL" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x1028" name="DATA_SIZE2">
      <bitfield name="RESERVED_0" low="11" high="31" type="uint"/>
      <bitfield name="DATAOUT_WIDTH" low="0" high="10" type="uint"/>
   </reg32>
   <reg32 offset="0x102C" name="DATA_SIZE3">
      <bitfield name="RESERVED_0" low="24" high="31" type="uint"/>
      <bitfield name="SURF_MODE" low="22" high="23" type="uint"/>
      <bitfield name="DATAOUT_ATOMICS" low="0" high="21" type="uint"/>
   </reg32>
   <reg32 offset="0x1030" name="WEIGHT_SIZE0">
      <bitfield name="WEIGHT_BYTES" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1034" name="WEIGHT_SIZE1">
      <bitfield name="RESERVED_0" low="19" high="31" type="uint"/>
      <bitfield name="WEIGHT_BYTES_PER_KERNEL" low="0" high="18" type="uint"/>
   </reg32>
   <reg32 offset="0x1038" name="WEIGHT_SIZE2">
      <bitfield name="RESERVED_0" low="29" high="31" type="uint"/>
      <bitfield name="WEIGHT_WIDTH" low="24" high="28" type="uint"/>
      <bitfield name="RESERVED_1" low="21" high="23" type="uint"/>
      <bitfield name="WEIGHT_HEIGHT" low="16" high="20" type="uint"/>
      <bitfield name="RESERVED_2" low="14" high="15" type="uint"/>
      <bitfield name="WEIGHT_KERNELS" low="0" high="13" type="uint"/>
   </reg32>
   <reg32 offset="0x1040" name="CBUF_CON0">
      <bitfield name="RESERVED_0" low="14" high="31" type="uint"/>
      <bitfield name="WEIGHT_REUSE" pos="13" type="uint"/>
      <bitfield name="DATA_REUSE" pos="12" type="uint"/>
      <bitfield name="RESERVED_1" pos="11" type="uint"/>
      <bitfield name="FC_DATA_BANK" low="8" high="10" type="uint"/>
      <bitfield name="WEIGHT_BANK" low="4" high="7" type="uint"/>
      <bitfield name="DATA_BANK" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x1044" name="CBUF_CON1">
      <bitfield name="RESERVED_0" low="14" high="31" type="uint"/>
      <bitfield name="DATA_ENTRIES" low="0" high="13" type="uint"/>
   </reg32>
   <reg32 offset="0x104C" name="CVT_CON0">
      <bitfield name="RESERVED_0" low="28" high="31" type="uint"/>
      <bitfield name="CVT_TRUNCATE_3" low="22" high="27" type="uint"/>
      <bitfield name="CVT_TRUNCATE_2" low="16" high="21" type="uint"/>
      <bitfield name="CVT_TRUNCATE_1" low="10" high="15" type="uint"/>
      <bitfield name="CVT_TRUNCATE_0" low="4" high="9" type="uint"/>
      <bitfield name="DATA_SIGN" pos="3" type="uint"/>
      <bitfield name="ROUND_TYPE" pos="2" type="uint"/>
      <bitfield name="CVT_TYPE" pos="1" type="uint"/>
      <bitfield name="CVT_BYPASS" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x1050" name="CVT_CON1">
      <bitfield name="CVT_SCALE0" low="16" high="31" type="uint"/>
      <bitfield name="CVT_OFFSET0" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x1054" name="CVT_CON2">
      <bitfield name="CVT_SCALE1" low="16" high="31" type="uint"/>
      <bitfield name="CVT_OFFSET1" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x1058" name="CVT_CON3">
      <bitfield name="CVT_SCALE2" low="16" high="31" type="uint"/>
      <bitfield name="CVT_OFFSET2" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x105C" name="CVT_CON4">
      <bitfield name="CVT_SCALE3" low="16" high="31" type="uint"/>
      <bitfield name="CVT_OFFSET3" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x1060" name="FC_CON0">
      <bitfield name="FC_SKIP_DATA" low="16" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="1" high="15" type="uint"/>
      <bitfield name="FC_SKIP_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x1064" name="FC_CON1">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="DATA_OFFSET" low="0" high="16" type="uint"/>
   </reg32>
   <reg32 offset="0x1068" name="PAD_CON0">
      <bitfield name="RESERVED_0" low="8" high="31" type="uint"/>
      <bitfield name="PAD_LEFT" low="4" high="7" type="uint"/>
      <bitfield name="PAD_TOP" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x1070" name="FEATURE_DATA_ADDR">
      <bitfield name="FEATURE_BASE_ADDR" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1074" name="FC_CON2">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="WEIGHT_OFFSET" low="0" high="16" type="uint"/>
   </reg32>
   <reg32 offset="0x1078" name="DMA_CON0">
      <bitfield name="OV4K_BYPASS" pos="31" type="uint"/>
      <bitfield name="RESERVED_0" low="20" high="30" type="uint"/>
      <bitfield name="WEIGHT_BURST_LEN" low="16" high="19" type="uint"/>
      <bitfield name="RESERVED_1" low="4" high="15" type="uint"/>
      <bitfield name="DATA_BURST_LEN" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x107C" name="DMA_CON1">
      <bitfield name="RESERVED_0" low="28" high="31" type="uint"/>
      <bitfield name="LINE_STRIDE" low="0" high="27" type="uint"/>
   </reg32>
   <reg32 offset="0x1080" name="DMA_CON2">
      <bitfield name="RESERVED_0" low="28" high="31" type="uint"/>
      <bitfield name="SURF_STRIDE" low="0" high="27" type="uint"/>
   </reg32>
   <reg32 offset="0x1084" name="FC_DATA_SIZE0">
      <bitfield name="RESERVED_0" low="30" high="31" type="uint"/>
      <bitfield name="DMA_WIDTH" low="16" high="29" type="uint"/>
      <bitfield name="RESERVED_1" low="11" high="15" type="uint"/>
      <bitfield name="DMA_HEIGHT" low="0" high="10" type="uint"/>
   </reg32>
   <reg32 offset="0x1088" name="FC_DATA_SIZE1">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="DMA_CHANNEL" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x1090" name="CLK_GATE">
      <bitfield name="RESERVED_0" low="5" high="31" type="uint"/>
      <bitfield name="CBUF_CS_DISABLE_CLKGATE" pos="4" type="uint"/>
      <bitfield name="RESERVED_1" pos="3" type="uint"/>
      <bitfield name="CSC_DISABLE_CLKGATE" pos="2" type="uint"/>
      <bitfield name="CNA_WEIGHT_DISABLE_CLKGATE" pos="1" type="uint"/>
      <bitfield name="CNA_FEATURE_DISABLE_CLKGATE" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x1100" name="DCOMP_CTRL">
      <bitfield name="RESERVED_0" low="4" high="31" type="uint"/>
      <bitfield name="WT_DEC_BYPASS" pos="3" type="uint"/>
      <bitfield name="DECOMP_CONTROL" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x1104" name="DCOMP_REGNUM">
      <bitfield name="DCOMP_REGNUM" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1110" name="DCOMP_ADDR0">
      <bitfield name="DECOMPRESS_ADDR0" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1140" name="DCOMP_AMOUNT0">
      <bitfield name="DCOMP_AMOUNT0" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1144" name="DCOMP_AMOUNT1">
      <bitfield name="DCOMP_AMOUNT1" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1148" name="DCOMP_AMOUNT2">
      <bitfield name="DCOMP_AMOUNT2" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x114C" name="DCOMP_AMOUNT3">
      <bitfield name="DCOMP_AMOUNT3" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1150" name="DCOMP_AMOUNT4">
      <bitfield name="DCOMP_AMOUNT4" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1154" name="DCOMP_AMOUNT5">
      <bitfield name="DCOMP_AMOUNT5" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1158" name="DCOMP_AMOUNT6">
      <bitfield name="DCOMP_AMOUNT6" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x115C" name="DCOMP_AMOUNT7">
      <bitfield name="DCOMP_AMOUNT7" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1160" name="DCOMP_AMOUNT8">
      <bitfield name="DCOMP_AMOUNT8" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1164" name="DCOMP_AMOUNT9">
      <bitfield name="DCOMP_AMOUNT9" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1168" name="DCOMP_AMOUNT10">
      <bitfield name="DCOMP_AMOUNT10" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x116C" name="DCOMP_AMOUNT11">
      <bitfield name="DCOMP_AMOUNT11" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1170" name="DCOMP_AMOUNT12">
      <bitfield name="DCOMP_AMOUNT12" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1174" name="DCOMP_AMOUNT13">
      <bitfield name="DCOMP_AMOUNT13" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1178" name="DCOMP_AMOUNT14">
      <bitfield name="DCOMP_AMOUNT14" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x117C" name="DCOMP_AMOUNT15">
      <bitfield name="DCOMP_AMOUNT15" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1180" name="CVT_CON5">
      <bitfield name="PER_CHANNEL_CVT_EN" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x1184" name="PAD_CON1">
      <bitfield name="PAD_VALUE" low="0" high="31" type="uint"/>
   </reg32>
</domain>
<domain name="CORE" width="32">
   <reg32 offset="0x3000" name="S_STATUS">
      <bitfield name="RESERVED_0" low="18" high="31" type="uint"/>
      <bitfield name="STATUS_1" low="16" high="17" type="uint"/>
      <bitfield name="RESERVED_1" low="2" high="15" type="uint"/>
      <bitfield name="STATUS_0" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x3004" name="S_POINTER">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="EXECUTER" pos="16" type="uint"/>
      <bitfield name="RESERVED_1" low="6" high="15" type="uint"/>
      <bitfield name="EXECUTER_PP_CLEAR" pos="5" type="uint"/>
      <bitfield name="POINTER_PP_CLEAR" pos="4" type="uint"/>
      <bitfield name="POINTER_PP_MODE" pos="3" type="uint"/>
      <bitfield name="EXECUTER_PP_EN" pos="2" type="uint"/>
      <bitfield name="POINTER_PP_EN" pos="1" type="uint"/>
      <bitfield name="POINTER" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x3008" name="OPERATION_ENABLE">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="OP_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x300C" name="MAC_GATING">
      <bitfield name="RESERVED_0" low="27" high="31" type="uint"/>
      <bitfield name="SLCG_OP_EN" low="0" high="26" type="uint"/>
   </reg32>
   <reg32 offset="0x3010" name="MISC_CFG">
      <bitfield name="RESERVED_0" low="20" high="31" type="uint"/>
      <bitfield name="SOFT_GATING" low="14" high="19" type="uint"/>
      <bitfield name="RESERVED_1" low="11" high="13" type="uint"/>
      <bitfield name="PROC_PRECISION" low="8" high="10" type="uint"/>
      <bitfield name="RESERVED_2" low="2" high="7" type="uint"/>
      <bitfield name="DW_EN" pos="1" type="uint"/>
      <bitfield name="QD_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x3014" name="DATAOUT_SIZE_0">
      <bitfield name="DATAOUT_HEIGHT" low="16" high="31" type="uint"/>
      <bitfield name="DATAOUT_WIDTH" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x3018" name="DATAOUT_SIZE_1">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="DATAOUT_CHANNEL" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x301C" name="CLIP_TRUNCATE">
      <bitfield name="RESERVED_0" low="7" high="31" type="uint"/>
      <bitfield name="ROUND_TYPE" pos="6" type="uint"/>
      <bitfield name="RESERVED_1" pos="5" type="uint"/>
      <bitfield name="CLIP_TRUNCATE" low="0" high="4" type="uint"/>
   </reg32>
</domain>
<domain name="DPU" width="32">
   <reg32 offset="0x4000" name="S_STATUS">
      <bitfield name="RESERVED_0" low="18" high="31" type="uint"/>
      <bitfield name="STATUS_1" low="16" high="17" type="uint"/>
      <bitfield name="RESERVED_1" low="2" high="15" type="uint"/>
      <bitfield name="STATUS_0" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x4004" name="S_POINTER">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="EXECUTER" pos="16" type="uint"/>
      <bitfield name="RESERVED_1" low="6" high="15" type="uint"/>
      <bitfield name="EXECUTER_PP_CLEAR" pos="5" type="uint"/>
      <bitfield name="POINTER_PP_CLEAR" pos="4" type="uint"/>
      <bitfield name="POINTER_PP_MODE" pos="3" type="uint"/>
      <bitfield name="EXECUTER_PP_EN" pos="2" type="uint"/>
      <bitfield name="POINTER_PP_EN" pos="1" type="uint"/>
      <bitfield name="POINTER" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x4008" name="OPERATION_ENABLE">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="OP_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x400C" name="FEATURE_MODE_CFG">
      <bitfield name="COMB_USE" pos="31" type="uint"/>
      <bitfield name="TP_EN" pos="30" type="uint"/>
      <bitfield name="RGP_TYPE" low="26" high="29" type="uint"/>
      <bitfield name="NONALIGN" pos="25" type="uint"/>
      <bitfield name="SURF_LEN" low="9" high="24" type="uint"/>
      <bitfield name="BURST_LEN" low="5" high="8" type="uint"/>
      <bitfield name="CONV_MODE" low="3" high="4" type="uint"/>
      <bitfield name="OUTPUT_MODE" low="1" high="2" type="uint"/>
      <bitfield name="FLYING_MODE" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x4010" name="DATA_FORMAT">
      <bitfield name="OUT_PRECISION" low="29" high="31" type="uint"/>
      <bitfield name="IN_PRECISION" low="26" high="28" type="uint"/>
      <bitfield name="EW_TRUNCATE_NEG" low="16" high="25" type="uint"/>
      <bitfield name="BN_MUL_SHIFT_VALUE_NEG" low="10" high="15" type="uint"/>
      <bitfield name="BS_MUL_SHIFT_VALUE_NEG" low="4" high="9" type="uint"/>
      <bitfield name="MC_SURF_OUT" pos="3" type="uint"/>
      <bitfield name="PROC_PRECISION" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x4014" name="OFFSET_PEND">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="OFFSET_PEND" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x4020" name="DST_BASE_ADDR">
      <bitfield name="DST_BASE_ADDR" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4024" name="DST_SURF_STRIDE">
      <bitfield name="DST_SURF_STRIDE" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x4030" name="DATA_CUBE_WIDTH">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="WIDTH" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x4034" name="DATA_CUBE_HEIGHT">
      <bitfield name="RESERVED_0" low="25" high="31" type="uint"/>
      <bitfield name="MINMAX_CTL" low="22" high="24" type="uint"/>
      <bitfield name="RESERVED_1" low="13" high="21" type="uint"/>
      <bitfield name="HEIGHT" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x4038" name="DATA_CUBE_NOTCH_ADDR">
      <bitfield name="RESERVED_0" low="29" high="31" type="uint"/>
      <bitfield name="NOTCH_ADDR_1" low="16" high="28" type="uint"/>
      <bitfield name="RESERVED_1" low="13" high="15" type="uint"/>
      <bitfield name="NOTCH_ADDR_0" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x403C" name="DATA_CUBE_CHANNEL">
      <bitfield name="RESERVED_0" low="29" high="31" type="uint"/>
      <bitfield name="ORIG_CHANNEL" low="16" high="28" type="uint"/>
      <bitfield name="RESERVED_1" low="13" high="15" type="uint"/>
      <bitfield name="CHANNEL" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x4040" name="BS_CFG">
      <bitfield name="RESERVED_0" low="20" high="31" type="uint"/>
      <bitfield name="BS_ALU_ALGO" low="16" high="19" type="uint"/>
      <bitfield name="RESERVED_1" low="9" high="15" type="uint"/>
      <bitfield name="BS_ALU_SRC" pos="8" type="uint"/>
      <bitfield name="BS_RELUX_EN" pos="7" type="uint"/>
      <bitfield name="BS_RELU_BYPASS" pos="6" type="uint"/>
      <bitfield name="BS_MUL_PRELU" pos="5" type="uint"/>
      <bitfield name="BS_MUL_BYPASS" pos="4" type="uint"/>
      <bitfield name="RESERVED_2" low="2" high="3" type="uint"/>
      <bitfield name="BS_ALU_BYPASS" pos="1" type="uint"/>
      <bitfield name="BS_BYPASS" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x4044" name="BS_ALU_CFG">
      <bitfield name="BS_ALU_OPERAND" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4048" name="BS_MUL_CFG">
      <bitfield name="BS_MUL_OPERAND" low="16" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="14" high="15" type="uint"/>
      <bitfield name="BS_MUL_SHIFT_VALUE" low="8" high="13" type="uint"/>
      <bitfield name="RESERVED_1" low="2" high="7" type="uint"/>
      <bitfield name="BS_TRUNCATE_SRC" pos="1" type="uint"/>
      <bitfield name="BS_MUL_SRC" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x404C" name="BS_RELUX_CMP_VALUE">
      <bitfield name="BS_RELUX_CMP_DAT" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4050" name="BS_OW_CFG">
      <bitfield name="RGP_CNTER" low="28" high="31" type="uint"/>
      <bitfield name="TP_ORG_EN" pos="27" type="uint"/>
      <bitfield name="RESERVED_0" low="11" high="26" type="uint"/>
      <bitfield name="SIZE_E_2" low="8" high="10" type="uint"/>
      <bitfield name="SIZE_E_1" low="5" high="7" type="uint"/>
      <bitfield name="SIZE_E_0" low="2" high="4" type="uint"/>
      <bitfield name="OD_BYPASS" pos="1" type="uint"/>
      <bitfield name="OW_SRC" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x4054" name="BS_OW_OP">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="OW_OP" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x4058" name="WDMA_SIZE_0">
      <bitfield name="RESERVED_0" low="28" high="31" type="uint"/>
      <bitfield name="TP_PRECISION" pos="27" type="uint"/>
      <bitfield name="SIZE_C_WDMA" low="16" high="26" type="uint"/>
      <bitfield name="RESERVED_1" low="13" high="15" type="uint"/>
      <bitfield name="CHANNEL_WDMA" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x405C" name="WDMA_SIZE_1">
      <bitfield name="RESERVED_0" low="29" high="31" type="uint"/>
      <bitfield name="HEIGHT_WDMA" low="16" high="28" type="uint"/>
      <bitfield name="RESERVED_1" low="13" high="15" type="uint"/>
      <bitfield name="WIDTH_WDMA" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x4060" name="BN_CFG">
      <bitfield name="RESERVED_0" low="20" high="31" type="uint"/>
      <bitfield name="BN_ALU_ALGO" low="16" high="19" type="uint"/>
      <bitfield name="RESERVED_1" low="9" high="15" type="uint"/>
      <bitfield name="BN_ALU_SRC" pos="8" type="uint"/>
      <bitfield name="BN_RELUX_EN" pos="7" type="uint"/>
      <bitfield name="BN_RELU_BYPASS" pos="6" type="uint"/>
      <bitfield name="BN_MUL_PRELU" pos="5" type="uint"/>
      <bitfield name="BN_MUL_BYPASS" pos="4" type="uint"/>
      <bitfield name="RESERVED_2" low="2" high="3" type="uint"/>
      <bitfield name="BN_ALU_BYPASS" pos="1" type="uint"/>
      <bitfield name="BN_BYPASS" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x4064" name="BN_ALU_CFG">
      <bitfield name="BN_ALU_OPERAND" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4068" name="BN_MUL_CFG">
      <bitfield name="BN_MUL_OPERAND" low="16" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="14" high="15" type="uint"/>
      <bitfield name="BN_MUL_SHIFT_VALUE" low="8" high="13" type="uint"/>
      <bitfield name="RESERVED_1" low="2" high="7" type="uint"/>
      <bitfield name="BN_TRUNCATE_SRC" pos="1" type="uint"/>
      <bitfield name="BN_MUL_SRC" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x406C" name="BN_RELUX_CMP_VALUE">
      <bitfield name="BN_RELUX_CMP_DAT" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4070" name="EW_CFG">
      <bitfield name="EW_CVT_TYPE" pos="31" type="uint"/>
      <bitfield name="EW_CVT_ROUND" pos="30" type="uint"/>
      <bitfield name="EW_DATA_MODE" low="28" high="29" type="uint"/>
      <bitfield name="RESERVED_0" low="24" high="27" type="uint"/>
      <bitfield name="EDATA_SIZE" low="22" high="23" type="uint"/>
      <bitfield name="EW_EQUAL_EN" pos="21" type="uint"/>
      <bitfield name="EW_BINARY_EN" pos="20" type="uint"/>
      <bitfield name="EW_ALU_ALGO" low="16" high="19" type="uint"/>
      <bitfield name="RESERVED_1" low="11" high="15" type="uint"/>
      <bitfield name="EW_RELUX_EN" pos="10" type="uint"/>
      <bitfield name="EW_RELU_BYPASS" pos="9" type="uint"/>
      <bitfield name="EW_OP_CVT_BYPASS" pos="8" type="uint"/>
      <bitfield name="EW_LUT_BYPASS" pos="7" type="uint"/>
      <bitfield name="EW_OP_SRC" pos="6" type="uint"/>
      <bitfield name="EW_MUL_PRELU" pos="5" type="uint"/>
      <bitfield name="RESERVED_2" low="3" high="4" type="uint"/>
      <bitfield name="EW_OP_TYPE" pos="2" type="uint"/>
      <bitfield name="EW_OP_BYPASS" pos="1" type="uint"/>
      <bitfield name="EW_BYPASS" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x4074" name="EW_CVT_OFFSET_VALUE">
      <bitfield name="EW_OP_CVT_OFFSET" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4078" name="EW_CVT_SCALE_VALUE">
      <bitfield name="EW_TRUNCATE" low="22" high="31" type="uint"/>
      <bitfield name="EW_OP_CVT_SHIFT" low="16" high="21" type="uint"/>
      <bitfield name="EW_OP_CVT_SCALE" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x407C" name="EW_RELUX_CMP_VALUE">
      <bitfield name="EW_RELUX_CMP_DAT" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4080" name="OUT_CVT_OFFSET">
      <bitfield name="OUT_CVT_OFFSET" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4084" name="OUT_CVT_SCALE">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="FP32TOFP16_EN" pos="16" type="uint"/>
      <bitfield name="OUT_CVT_SCALE" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x4088" name="OUT_CVT_SHIFT">
      <bitfield name="CVT_TYPE" pos="31" type="uint"/>
      <bitfield name="CVT_ROUND" pos="30" type="uint"/>
      <bitfield name="RESERVED_0" low="20" high="29" type="uint"/>
      <bitfield name="MINUS_EXP" low="12" high="19" type="uint"/>
      <bitfield name="OUT_CVT_SHIFT" low="0" high="11" type="uint"/>
   </reg32>
   <reg32 offset="0x4090" name="EW_OP_VALUE_0">
      <bitfield name="EW_OPERAND_0" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4094" name="EW_OP_VALUE_1">
      <bitfield name="EW_OPERAND_1" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4098" name="EW_OP_VALUE_2">
      <bitfield name="EW_OPERAND_2" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x409C" name="EW_OP_VALUE_3">
      <bitfield name="EW_OPERAND_3" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x40A0" name="EW_OP_VALUE_4">
      <bitfield name="EW_OPERAND_4" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x40A4" name="EW_OP_VALUE_5">
      <bitfield name="EW_OPERAND_5" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x40A8" name="EW_OP_VALUE_6">
      <bitfield name="EW_OPERAND_6" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x40AC" name="EW_OP_VALUE_7">
      <bitfield name="EW_OPERAND_7" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x40C0" name="SURFACE_ADD">
      <bitfield name="SURF_ADD" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x4100" name="LUT_ACCESS_CFG">
      <bitfield name="RESERVED_0" low="18" high="31" type="uint"/>
      <bitfield name="LUT_ACCESS_TYPE" pos="17" type="uint"/>
      <bitfield name="LUT_TABLE_ID" pos="16" type="uint"/>
      <bitfield name="RESERVED_1" low="10" high="15" type="uint"/>
      <bitfield name="LUT_ADDR" low="0" high="9" type="uint"/>
   </reg32>
   <reg32 offset="0x4104" name="LUT_ACCESS_DATA">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="LUT_ACCESS_DATA" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x4108" name="LUT_CFG">
      <bitfield name="RESERVED_0" low="8" high="31" type="uint"/>
      <bitfield name="LUT_CAL_SEL" pos="7" type="uint"/>
      <bitfield name="LUT_HYBRID_PRIORITY" pos="6" type="uint"/>
      <bitfield name="LUT_OFLOW_PRIORITY" pos="5" type="uint"/>
      <bitfield name="LUT_UFLOW_PRIORITY" pos="4" type="uint"/>
      <bitfield name="LUT_LO_LE_MUX" low="2" high="3" type="uint"/>
      <bitfield name="LUT_EXPAND_EN" pos="1" type="uint"/>
      <bitfield name="LUT_ROAD_SEL" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x410C" name="LUT_INFO">
      <bitfield name="RESERVED_0" low="24" high="31" type="uint"/>
      <bitfield name="LUT_LO_INDEX_SELECT" low="16" high="23" type="uint"/>
      <bitfield name="LUT_LE_INDEX_SELECT" low="8" high="15" type="uint"/>
      <bitfield name="RESERVED_1" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x4110" name="LUT_LE_START">
      <bitfield name="LUT_LE_START" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4114" name="LUT_LE_END">
      <bitfield name="LUT_LE_END" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4118" name="LUT_LO_START">
      <bitfield name="LUT_LO_START" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x411C" name="LUT_LO_END">
      <bitfield name="LUT_LO_END" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x4120" name="LUT_LE_SLOPE_SCALE">
      <bitfield name="LUT_LE_SLOPE_OFLOW_SCALE" low="16" high="31" type="uint"/>
      <bitfield name="LUT_LE_SLOPE_UFLOW_SCALE" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x4124" name="LUT_LE_SLOPE_SHIFT">
      <bitfield name="RESERVED_0" low="10" high="31" type="uint"/>
      <bitfield name="LUT_LE_SLOPE_OFLOW_SHIFT" low="5" high="9" type="uint"/>
      <bitfield name="LUT_LE_SLOPE_UFLOW_SHIFT" low="0" high="4" type="uint"/>
   </reg32>
   <reg32 offset="0x4128" name="LUT_LO_SLOPE_SCALE">
      <bitfield name="LUT_LO_SLOPE_OFLOW_SCALE" low="16" high="31" type="uint"/>
      <bitfield name="LUT_LO_SLOPE_UFLOW_SCALE" low="0" high="15" type="uint"/>
   </reg32>
   <reg32 offset="0x412C" name="LUT_LO_SLOPE_SHIFT">
      <bitfield name="RESERVED_0" low="10" high="31" type="uint"/>
      <bitfield name="LUT_LO_SLOPE_OFLOW_SHIFT" low="5" high="9" type="uint"/>
      <bitfield name="LUT_LO_SLOPE_UFLOW_SHIFT" low="0" high="4" type="uint"/>
   </reg32>
</domain>
<domain name="DPU_RDMA" width="32">
   <reg32 offset="0x5000" name="RDMA_S_STATUS">
      <bitfield name="RESERVED_0" low="18" high="31" type="uint"/>
      <bitfield name="STATUS_1" low="16" high="17" type="uint"/>
      <bitfield name="RESERVED_1" low="2" high="15" type="uint"/>
      <bitfield name="STATUS_0" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x5004" name="RDMA_S_POINTER">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="EXECUTER" pos="16" type="uint"/>
      <bitfield name="RESERVED_1" low="6" high="15" type="uint"/>
      <bitfield name="EXECUTER_PP_CLEAR" pos="5" type="uint"/>
      <bitfield name="POINTER_PP_CLEAR" pos="4" type="uint"/>
      <bitfield name="POINTER_PP_MODE" pos="3" type="uint"/>
      <bitfield name="EXECUTER_PP_EN" pos="2" type="uint"/>
      <bitfield name="POINTER_PP_EN" pos="1" type="uint"/>
      <bitfield name="POINTER" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x5008" name="RDMA_OPERATION_ENABLE">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="OP_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x500C" name="RDMA_DATA_CUBE_WIDTH">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="WIDTH" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x5010" name="RDMA_DATA_CUBE_HEIGHT">
      <bitfield name="RESERVED_0" low="29" high="31" type="uint"/>
      <bitfield name="EW_LINE_NOTCH_ADDR" low="16" high="28" type="uint"/>
      <bitfield name="RESERVED_1" low="13" high="15" type="uint"/>
      <bitfield name="HEIGHT" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x5014" name="RDMA_DATA_CUBE_CHANNEL">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CHANNEL" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x5018" name="RDMA_SRC_BASE_ADDR">
      <bitfield name="SRC_BASE_ADDR" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x501C" name="RDMA_BRDMA_CFG">
      <bitfield name="RESERVED_0" low="5" high="31" type="uint"/>
      <bitfield name="BRDMA_DATA_USE" low="1" high="4" type="uint"/>
      <bitfield name="RESERVED_1" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x5020" name="RDMA_BS_BASE_ADDR">
      <bitfield name="BS_BASE_ADDR" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x5028" name="RDMA_NRDMA_CFG">
      <bitfield name="RESERVED_0" low="5" high="31" type="uint"/>
      <bitfield name="NRDMA_DATA_USE" low="1" high="4" type="uint"/>
      <bitfield name="RESERVED_1" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x502C" name="RDMA_BN_BASE_ADDR">
      <bitfield name="BN_BASE_ADDR" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x5034" name="RDMA_ERDMA_CFG">
      <bitfield name="ERDMA_DATA_MODE" low="30" high="31" type="uint"/>
      <bitfield name="ERDMA_SURF_MODE" pos="29" type="uint"/>
      <bitfield name="ERDMA_NONALIGN" pos="28" type="uint"/>
      <bitfield name="RESERVED_0" low="4" high="27" type="uint"/>
      <bitfield name="ERDMA_DATA_SIZE" low="2" high="3" type="uint"/>
      <bitfield name="OV4K_BYPASS" pos="1" type="uint"/>
      <bitfield name="ERDMA_DISABLE" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x5038" name="RDMA_EW_BASE_ADDR">
      <bitfield name="EW_BASE_ADDR" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x5040" name="RDMA_EW_SURF_STRIDE">
      <bitfield name="EW_SURF_STRIDE" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x5044" name="RDMA_FEATURE_MODE_CFG">
      <bitfield name="RESERVED_0" low="18" high="31" type="uint"/>
      <bitfield name="IN_PRECISION" low="15" high="17" type="uint"/>
      <bitfield name="BURST_LEN" low="11" high="14" type="uint"/>
      <bitfield name="COMB_USE" low="8" high="10" type="uint"/>
      <bitfield name="PROC_PRECISION" low="5" high="7" type="uint"/>
      <bitfield name="MRDMA_DISABLE" pos="4" type="uint"/>
      <bitfield name="MRDMA_FP16TOFP32_EN" pos="3" type="uint"/>
      <bitfield name="CONV_MODE" low="1" high="2" type="uint"/>
      <bitfield name="FLYING_MODE" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x5048" name="RDMA_SRC_DMA_CFG">
      <bitfield name="LINE_NOTCH_ADDR" low="19" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="14" high="18" type="uint"/>
      <bitfield name="POOLING_METHOD" pos="13" type="uint"/>
      <bitfield name="UNPOOLING_EN" pos="12" type="uint"/>
      <bitfield name="KERNEL_STRIDE_HEIGHT" low="9" high="11" type="uint"/>
      <bitfield name="KERNEL_STRIDE_WIDTH" low="6" high="8" type="uint"/>
      <bitfield name="KERNEL_HEIGHT" low="3" high="5" type="uint"/>
      <bitfield name="KERNEL_WIDTH" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x504C" name="RDMA_SURF_NOTCH">
      <bitfield name="SURF_NOTCH_ADDR" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x5064" name="RDMA_PAD_CFG">
      <bitfield name="PAD_VALUE" low="16" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="7" high="15" type="uint"/>
      <bitfield name="PAD_TOP" low="4" high="6" type="uint"/>
      <bitfield name="RESERVED_1" pos="3" type="uint"/>
      <bitfield name="PAD_LEFT" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x5068" name="RDMA_WEIGHT">
      <bitfield name="E_WEIGHT" low="24" high="31" type="uint"/>
      <bitfield name="N_WEIGHT" low="16" high="23" type="uint"/>
      <bitfield name="B_WEIGHT" low="8" high="15" type="uint"/>
      <bitfield name="M_WEIGHT" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x506C" name="RDMA_EW_SURF_NOTCH">
      <bitfield name="EW_SURF_NOTCH" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
</domain>
<domain name="PPU" width="32">
   <reg32 offset="0x6000" name="S_STATUS">
      <bitfield name="RESERVED_0" low="18" high="31" type="uint"/>
      <bitfield name="STATUS_1" low="16" high="17" type="uint"/>
      <bitfield name="RESERVED_1" low="2" high="15" type="uint"/>
      <bitfield name="STATUS_0" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x6004" name="S_POINTER">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="EXECUTER" pos="16" type="uint"/>
      <bitfield name="RESERVED_1" low="6" high="15" type="uint"/>
      <bitfield name="EXECUTER_PP_CLEAR" pos="5" type="uint"/>
      <bitfield name="POINTER_PP_CLEAR" pos="4" type="uint"/>
      <bitfield name="POINTER_PP_MODE" pos="3" type="uint"/>
      <bitfield name="EXECUTER_PP_EN" pos="2" type="uint"/>
      <bitfield name="POINTER_PP_EN" pos="1" type="uint"/>
      <bitfield name="POINTER" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x6008" name="OPERATION_ENABLE">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="OP_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x600C" name="DATA_CUBE_IN_WIDTH">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_IN_WIDTH" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x6010" name="DATA_CUBE_IN_HEIGHT">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_IN_HEIGHT" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x6014" name="DATA_CUBE_IN_CHANNEL">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_IN_CHANNEL" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x6018" name="DATA_CUBE_OUT_WIDTH">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_OUT_WIDTH" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x601C" name="DATA_CUBE_OUT_HEIGHT">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_OUT_HEIGHT" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x6020" name="DATA_CUBE_OUT_CHANNEL">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_OUT_CHANNEL" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x6024" name="OPERATION_MODE_CFG">
      <bitfield name="RESERVED_0" pos="31" type="uint"/>
      <bitfield name="INDEX_EN" pos="30" type="uint"/>
      <bitfield name="RESERVED_1" pos="29" type="uint"/>
      <bitfield name="NOTCH_ADDR" low="16" high="28" type="uint"/>
      <bitfield name="RESERVED_2" low="8" high="15" type="uint"/>
      <bitfield name="USE_CNT" low="5" high="7" type="uint"/>
      <bitfield name="FLYING_MODE" pos="4" type="uint"/>
      <bitfield name="RESERVED_3" low="2" high="3" type="uint"/>
      <bitfield name="POOLING_METHOD" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x6034" name="POOLING_KERNEL_CFG">
      <bitfield name="RESERVED_0" low="24" high="31" type="uint"/>
      <bitfield name="KERNEL_STRIDE_HEIGHT" low="20" high="23" type="uint"/>
      <bitfield name="KERNEL_STRIDE_WIDTH" low="16" high="19" type="uint"/>
      <bitfield name="RESERVED_1" low="12" high="15" type="uint"/>
      <bitfield name="KERNEL_HEIGHT" low="8" high="11" type="uint"/>
      <bitfield name="RESERVED_2" low="4" high="7" type="uint"/>
      <bitfield name="KERNEL_WIDTH" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x6038" name="RECIP_KERNEL_WIDTH">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="RECIP_KERNEL_WIDTH" low="0" high="16" type="uint"/>
   </reg32>
   <reg32 offset="0x603C" name="RECIP_KERNEL_HEIGHT">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="RECIP_KERNEL_HEIGHT" low="0" high="16" type="uint"/>
   </reg32>
   <reg32 offset="0x6040" name="POOLING_PADDING_CFG">
      <bitfield name="RESERVED_0" low="15" high="31" type="uint"/>
      <bitfield name="PAD_BOTTOM" low="12" high="14" type="uint"/>
      <bitfield name="RESERVED_1" pos="11" type="uint"/>
      <bitfield name="PAD_RIGHT" low="8" high="10" type="uint"/>
      <bitfield name="RESERVED_2" pos="7" type="uint"/>
      <bitfield name="PAD_TOP" low="4" high="6" type="uint"/>
      <bitfield name="RESERVED_3" pos="3" type="uint"/>
      <bitfield name="PAD_LEFT" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x6044" name="PADDING_VALUE_1_CFG">
      <bitfield name="PAD_VALUE_0" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x6048" name="PADDING_VALUE_2_CFG">
      <bitfield name="RESERVED_0" low="3" high="31" type="uint"/>
      <bitfield name="PAD_VALUE_1" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x6070" name="DST_BASE_ADDR">
      <bitfield name="DST_BASE_ADDR" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x607C" name="DST_SURF_STRIDE">
      <bitfield name="DST_SURF_STRIDE" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x6084" name="DATA_FORMAT">
      <bitfield name="INDEX_ADD" low="4" high="31" type="uint"/>
      <bitfield name="DPU_FLYIN" pos="3" type="uint"/>
      <bitfield name="PROC_PRECISION" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x60DC" name="MISC_CTRL">
      <bitfield name="SURF_LEN" low="16" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="9" high="15" type="uint"/>
      <bitfield name="MC_SURF_OUT" pos="8" type="uint"/>
      <bitfield name="NONALIGN" pos="7" type="uint"/>
      <bitfield name="RESERVED_1" low="4" high="6" type="uint"/>
      <bitfield name="BURST_LEN" low="0" high="3" type="uint"/>
   </reg32>
</domain>
<domain name="PPU_RDMA" width="32">
   <reg32 offset="0x7000" name="RDMA_S_STATUS">
      <bitfield name="RESERVED_0" low="18" high="31" type="uint"/>
      <bitfield name="STATUS_1" low="16" high="17" type="uint"/>
      <bitfield name="RESERVED_1" low="2" high="15" type="uint"/>
      <bitfield name="STATUS_0" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x7004" name="RDMA_S_POINTER">
      <bitfield name="RESERVED_0" low="17" high="31" type="uint"/>
      <bitfield name="EXECUTER" pos="16" type="uint"/>
      <bitfield name="RESERVED_1" low="6" high="15" type="uint"/>
      <bitfield name="EXECUTER_PP_CLEAR" pos="5" type="uint"/>
      <bitfield name="POINTER_PP_CLEAR" pos="4" type="uint"/>
      <bitfield name="POINTER_PP_MODE" pos="3" type="uint"/>
      <bitfield name="EXECUTER_PP_EN" pos="2" type="uint"/>
      <bitfield name="POINTER_PP_EN" pos="1" type="uint"/>
      <bitfield name="POINTER" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x7008" name="RDMA_OPERATION_ENABLE">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="OP_EN" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x700C" name="RDMA_CUBE_IN_WIDTH">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_IN_WIDTH" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x7010" name="RDMA_CUBE_IN_HEIGHT">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_IN_HEIGHT" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x7014" name="RDMA_CUBE_IN_CHANNEL">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="CUBE_IN_CHANNEL" low="0" high="12" type="uint"/>
   </reg32>
   <reg32 offset="0x701C" name="RDMA_SRC_BASE_ADDR">
      <bitfield name="SRC_BASE_ADDR" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x7024" name="RDMA_SRC_LINE_STRIDE">
      <bitfield name="SRC_LINE_STRIDE" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x7028" name="RDMA_SRC_SURF_STRIDE">
      <bitfield name="SRC_SURF_STRIDE" low="4" high="31" type="uint"/>
      <bitfield name="RESERVED_0" low="0" high="3" type="uint"/>
   </reg32>
   <reg32 offset="0x7030" name="RDMA_DATA_FORMAT">
      <bitfield name="RESERVED_0" low="2" high="31" type="uint"/>
      <bitfield name="IN_PRECISION" low="0" high="1" type="uint"/>
   </reg32>
</domain>
<domain name="DDMA" width="32">
   <reg32 offset="0x8000" name="CFG_OUTSTANDING">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="WR_OS_CNT" low="8" high="15" type="uint"/>
      <bitfield name="RD_OS_CNT" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x8004" name="RD_WEIGHT_0">
      <bitfield name="RD_WEIGHT_PDP" low="24" high="31" type="uint"/>
      <bitfield name="RD_WEIGHT_DPU" low="16" high="23" type="uint"/>
      <bitfield name="RD_WEIGHT_KERNEL" low="8" high="15" type="uint"/>
      <bitfield name="RD_WEIGHT_FEATURE" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x8008" name="WR_WEIGHT_0">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="WR_WEIGHT_PDP" low="8" high="15" type="uint"/>
      <bitfield name="WR_WEIGHT_DPU" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x800C" name="CFG_ID_ERROR">
      <bitfield name="RESERVED_0" low="10" high="31" type="uint"/>
      <bitfield name="WR_RESP_ID" low="6" high="9" type="uint"/>
      <bitfield name="RESERVED_1" pos="5" type="uint"/>
      <bitfield name="RD_RESP_ID" low="0" high="4" type="uint"/>
   </reg32>
   <reg32 offset="0x8010" name="RD_WEIGHT_1">
      <bitfield name="RESERVED_0" low="8" high="31" type="uint"/>
      <bitfield name="RD_WEIGHT_PC" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x8014" name="CFG_DMA_FIFO_CLR">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="DMA_FIFO_CLR" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x8018" name="CFG_DMA_ARB">
      <bitfield name="RESERVED_0" low="10" high="31" type="uint"/>
      <bitfield name="WR_ARBIT_MODEL" pos="9" type="uint"/>
      <bitfield name="RD_ARBIT_MODEL" pos="8" type="uint"/>
      <bitfield name="RESERVED_1" pos="7" type="uint"/>
      <bitfield name="WR_FIX_ARB" low="4" high="6" type="uint"/>
      <bitfield name="RESERVED_2" pos="3" type="uint"/>
      <bitfield name="RD_FIX_ARB" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x8020" name="CFG_DMA_RD_QOS">
      <bitfield name="RESERVED_0" low="10" high="31" type="uint"/>
      <bitfield name="RD_PC_QOS" low="8" high="9" type="uint"/>
      <bitfield name="RD_PPU_QOS" low="6" high="7" type="uint"/>
      <bitfield name="RD_DPU_QOS" low="4" high="5" type="uint"/>
      <bitfield name="RD_KERNEL_QOS" low="2" high="3" type="uint"/>
      <bitfield name="RD_FEATURE_QOS" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x8024" name="CFG_DMA_RD_CFG">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="RD_ARLOCK" pos="12" type="uint"/>
      <bitfield name="RD_ARCACHE" low="8" high="11" type="uint"/>
      <bitfield name="RD_ARPROT" low="5" high="7" type="uint"/>
      <bitfield name="RD_ARBURST" low="3" high="4" type="uint"/>
      <bitfield name="RD_ARSIZE" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x8028" name="CFG_DMA_WR_CFG">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="WR_AWLOCK" pos="12" type="uint"/>
      <bitfield name="WR_AWCACHE" low="8" high="11" type="uint"/>
      <bitfield name="WR_AWPROT" low="5" high="7" type="uint"/>
      <bitfield name="WR_AWBURST" low="3" high="4" type="uint"/>
      <bitfield name="WR_AWSIZE" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x802C" name="CFG_DMA_WSTRB">
      <bitfield name="WR_WSTRB" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x8030" name="CFG_STATUS">
      <bitfield name="RESERVED_0" low="9" high="31" type="uint"/>
      <bitfield name="IDEL" pos="8" type="uint"/>
      <bitfield name="RESERVED_1" low="0" high="7" type="uint"/>
   </reg32>
</domain>
<domain name="SDMA" width="32">
   <reg32 offset="0x9000" name="CFG_OUTSTANDING">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="WR_OS_CNT" low="8" high="15" type="uint"/>
      <bitfield name="RD_OS_CNT" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x9004" name="RD_WEIGHT_0">
      <bitfield name="RD_WEIGHT_PDP" low="24" high="31" type="uint"/>
      <bitfield name="RD_WEIGHT_DPU" low="16" high="23" type="uint"/>
      <bitfield name="RD_WEIGHT_KERNEL" low="8" high="15" type="uint"/>
      <bitfield name="RD_WEIGHT_FEATURE" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x9008" name="WR_WEIGHT_0">
      <bitfield name="RESERVED_0" low="16" high="31" type="uint"/>
      <bitfield name="WR_WEIGHT_PDP" low="8" high="15" type="uint"/>
      <bitfield name="WR_WEIGHT_DPU" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x900C" name="CFG_ID_ERROR">
      <bitfield name="RESERVED_0" low="10" high="31" type="uint"/>
      <bitfield name="WR_RESP_ID" low="6" high="9" type="uint"/>
      <bitfield name="RESERVED_1" pos="5" type="uint"/>
      <bitfield name="RD_RESP_ID" low="0" high="4" type="uint"/>
   </reg32>
   <reg32 offset="0x9010" name="RD_WEIGHT_1">
      <bitfield name="RESERVED_0" low="8" high="31" type="uint"/>
      <bitfield name="RD_WEIGHT_PC" low="0" high="7" type="uint"/>
   </reg32>
   <reg32 offset="0x9014" name="CFG_DMA_FIFO_CLR">
      <bitfield name="RESERVED_0" low="1" high="31" type="uint"/>
      <bitfield name="DMA_FIFO_CLR" pos="0" type="uint"/>
   </reg32>
   <reg32 offset="0x9018" name="CFG_DMA_ARB">
      <bitfield name="RESERVED_0" low="10" high="31" type="uint"/>
      <bitfield name="WR_ARBIT_MODEL" pos="9" type="uint"/>
      <bitfield name="RD_ARBIT_MODEL" pos="8" type="uint"/>
      <bitfield name="RESERVED_1" pos="7" type="uint"/>
      <bitfield name="WR_FIX_ARB" low="4" high="6" type="uint"/>
      <bitfield name="RESERVED_2" pos="3" type="uint"/>
      <bitfield name="RD_FIX_ARB" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x9020" name="CFG_DMA_RD_QOS">
      <bitfield name="RESERVED_0" low="10" high="31" type="uint"/>
      <bitfield name="RD_PC_QOS" low="8" high="9" type="uint"/>
      <bitfield name="RD_PPU_QOS" low="6" high="7" type="uint"/>
      <bitfield name="RD_DPU_QOS" low="4" high="5" type="uint"/>
      <bitfield name="RD_KERNEL_QOS" low="2" high="3" type="uint"/>
      <bitfield name="RD_FEATURE_QOS" low="0" high="1" type="uint"/>
   </reg32>
   <reg32 offset="0x9024" name="CFG_DMA_RD_CFG">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="RD_ARLOCK" pos="12" type="uint"/>
      <bitfield name="RD_ARCACHE" low="8" high="11" type="uint"/>
      <bitfield name="RD_ARPROT" low="5" high="7" type="uint"/>
      <bitfield name="RD_ARBURST" low="3" high="4" type="uint"/>
      <bitfield name="RD_ARSIZE" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x9028" name="CFG_DMA_WR_CFG">
      <bitfield name="RESERVED_0" low="13" high="31" type="uint"/>
      <bitfield name="WR_AWLOCK" pos="12" type="uint"/>
      <bitfield name="WR_AWCACHE" low="8" high="11" type="uint"/>
      <bitfield name="WR_AWPROT" low="5" high="7" type="uint"/>
      <bitfield name="WR_AWBURST" low="3" high="4" type="uint"/>
      <bitfield name="WR_AWSIZE" low="0" high="2" type="uint"/>
   </reg32>
   <reg32 offset="0x902C" name="CFG_DMA_WSTRB">
      <bitfield name="WR_WSTRB" low="0" high="31" type="uint"/>
   </reg32>
   <reg32 offset="0x9030" name="CFG_STATUS">
      <bitfield name="RESERVED_0" low="9" high="31" type="uint"/>
      <bitfield name="IDEL" pos="8" type="uint"/>
      <bitfield name="RESERVED_1" low="0" high="7" type="uint"/>
   </reg32>
</domain>
<domain name="GLOBAL" width="32">
   <reg32 offset="0xF008" name="OPERATION_ENABLE">
      <bitfield name="RESERVED_0" low="7" high="31" type="uint"/>
      <bitfield name="PPU_RDMA_OP_EN" pos="6" type="uint"/>
      <bitfield name="PPU_OP_EN" pos="5" type="uint"/>
      <bitfield name="DPU_RDMA_OP_EN" pos="4" type="uint"/>
      <bitfield name="DPU_OP_EN" pos="3" type="uint"/>
      <bitfield name="CORE_OP_EN" pos="2" type="uint"/>
      <bitfield name="RESERVED_1" pos="1" type="uint"/>
      <bitfield name="CNA_OP_EN" pos="0" type="uint"/>
   </reg32>
</domain>

</database>

"""


def parse_xml():
    root = ET.fromstring(XML_DATA)
    domains = {}
    
    # 定义命名空间
    namespaces = {'ns': 'http://nouveau.freedesktop.org/'}

    for domain in root.findall(".//ns:domain", namespaces):
        domain_name = domain.get("name")
        domain_width = int(domain.get("width"))
        registers = {}

        for reg in domain.findall("ns:reg32", namespaces):
            offset = int(reg.get("offset"), 16)
            reg_name = reg.get("name")
            bitfields = {}

            for bitfield in reg.findall("ns:bitfield", namespaces):
                bf_name = bitfield.get("name")
                if "pos" in bitfield.attrib:
                    low = high = int(bitfield.get("pos"))
                else:
                    low = int(bitfield.get("low"))
                    high = int(bitfield.get("high"))
                bitfields[bf_name] = (low, high)

            registers[offset] = {
                "name": reg_name,
                "bitfields": bitfields
            }

        domains[domain_name] = {
            "width": domain_width,
            "registers": registers
        }

    return domains

def find_register(domains, offset):
    for domain_name, domain_info in domains.items():
        if offset in domain_info["registers"]:
            reg_info = domain_info["registers"][offset]
            return domain_name, reg_info
    return None, None

def parse_error_log(log_line):
    pattern = r"offset: (0x[0-9a-fA-F]+), shift = (\d+), limit: (0x[0-9a-fA-F]+), value: (0x[0-9a-fA-F]+)"
    match = re.search(pattern, log_line)
    if match:
        return {
            "offset": int(match.group(1), 16),
            "shift": int(match.group(2)),
            "limit": int(match.group(3), 16),
            "value": int(match.group(4), 16)
        }
    return None

def decode_error(domains, error_info):
    domain_name, reg_info = find_register(domains, error_info["offset"])
    if not reg_info:
        return "(Register not found)"

    for bf_name, (low, high) in reg_info["bitfields"].items():
        if low <= error_info["shift"] <= high:
            actual_value = error_info["value"]
            max_value = (1 << (high - low + 1)) - 1
            return f" {domain_name} {reg_info['name'].lower()}.{bf_name.lower()}: value={actual_value} > {max_value}"

    return "(Bitfield not found)"

def main():
    domains = parse_xml()
    # print(f"Parsed domains: {', '.join(domains.keys())}")

    for line in sys.stdin:
        error_info = parse_error_log(line.strip())
        if error_info:
            decoded_error = decode_error(domains, error_info)
            print(f"{line.strip()} //{decoded_error}")
        else:
            print(f"{line.strip()}")

if __name__ == "__main__":
    main()