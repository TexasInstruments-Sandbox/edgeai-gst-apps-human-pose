#!/bin/bash

################################################################################

NUM_BUFFERS=600
LOOP_COUNT=1
LOG_FILE="$SOC"_perf_stats.csv
START_DUMPS=300
NUM_DUMPS=5
PERF="queue ! tiperfoverlay dump=true overlay=false location=$LOG_FILE \
      start-dumps=$START_DUMPS num-dumps=$NUM_DUMPS"
FILTER=""

################################################################################
VIDEO_FILE_MP4_1MP=/opt/edgeai-test-data/videos/video_0000_h264.mp4
VIDEO_FILE_H264_1MP=/opt/edgeai-test-data/videos/video_0000_h264.h264
VIDEO_FILE_H264_2MP=/opt/edgeai-test-data/videos/video_0000_h264_2mp.h264
VIDEO_FILE_H265_2MP=/opt/edgeai-test-data/videos/video_0000_h265_2mp.h265

if [ "$SOC" == "j721e" ]
then
  H264_DECODE="v4l2h264dec capture-io-mode=5 ! tiovxmemalloc pool-size=8"
  H265_DECODE="v4l2h265dec capture-io-mode=5 ! tiovxmemalloc pool-size=8"
else
  H264_DECODE="v4l2h264dec"
  H265_DECODE="v4l2h265dec"
fi

VIDEO_H264_2MP()
{
  cp $VIDEO_FILE_H264_2MP $VIDEO_FILE_H264_2MP$1
  echo "multifilesrc location=$VIDEO_FILE_H264_2MP$1"
  echo "stop-index=$LOOP_COUNT"
  echo "caps=\"video/x-h264, width=1920, height=1080\" !"
  echo "h264parse ! $H264_DECODE ! video/x-raw,format=NV12"
}

VIDEO_H265_2MP()
{
  cp $VIDEO_FILE_H265_2MP $VIDEO_FILE_H265_2MP$1
  echo "multifilesrc location=$VIDEO_FILE_H265_2MP$1"
  echo "stop-index=$LOOP_COUNT"
  echo "caps=\"video/x-h265, width=1920, height=1088\" !"
  echo "h265parse ! $H265_DECODE ! video/x-raw,format=NV12"
}

################################################################################

IMX219_0_DEV=(`$EDGEAI_GST_APPS_PATH/scripts/setup_cameras.sh | grep "CSI Camera 0" -A 4 | grep device`)
IMX219_0_DEV=${IMX219_0_DEV[2]}
if [ "$IMX219_0_DEV" == "" ]
then
  echo "[WARN] IMX219 camera 0 not connected, Skipping tests"
fi

IMX219_0_SUBDEV=(`$EDGEAI_GST_APPS_PATH/scripts/setup_cameras.sh | grep "CSI Camera 0" -A 4 | grep subdev_id`)
IMX219_0_SUBDEV=${IMX219_0_SUBDEV[2]}

IMX219_1_DEV=(`$EDGEAI_GST_APPS_PATH/scripts/setup_cameras.sh | grep "CSI Camera 1" -A 4 | grep device`)
IMX219_1_DEV=${IMX219_1_DEV[2]}
if [ "$IMX219_1_DEV" == "" ]
then
  echo "[WARN] IMX219 camera 1 not connected, Skipping tests"
fi

IMX219_1_SUBDEV=(`$EDGEAI_GST_APPS_PATH/scripts/setup_cameras.sh | grep "CSI Camera 1" -A 4 | grep subdev_id`)
IMX219_1_SUBDEV=${IMX219_1_SUBDEV[2]}

IMX219_0_SRC="v4l2src device=$IMX219_0_DEV io-mode=5 num-buffers=$NUM_BUFFERS"
IMX219_1_SRC="v4l2src device=$IMX219_1_DEV io-mode=5 num-buffers=$NUM_BUFFERS"
IMX219_FMT="video/x-bayer, width=1920, height=1080, format=rggb"
IMX219_ISP_COMMON_PROPS="dcc-isp-file=/opt/imaging/imx219/dcc_viss.bin \
                         format-msb=7 \
                         sink_0::dcc-2a-file=/opt/imaging/imx219/dcc_2a.bin"
IMX219_0_ISP="tiovxisp $IMX219_ISP_COMMON_PROPS sink_0::device=$IMX219_0_SUBDEV"
IMX219_1_ISP="tiovxisp $IMX219_ISP_COMMON_PROPS sink_0::device=$IMX219_1_SUBDEV"
IMX219_0="$IMX219_0_SRC ! queue ! $IMX219_FMT ! $IMX219_0_ISP"
IMX219_1="$IMX219_1_SRC ! queue ! $IMX219_FMT ! $IMX219_1_ISP"

################################################################################

POST_PROC_PROPS="alpha=0.400000 viz-threshold=0.600000 top-N=5"
POST_PROC_CAPS="video/x-raw, width=1280, height=720"

MODEL_OD=/opt/model_zoo/TFL-OD-2020-ssdLite-mobDet-DSP-coco-320x320
MODEL_OD_PRE_PROC_PROPS="data-type=3 channel-order=1 tensor-format=rgb"
MODEL_OD_CAPS="video/x-raw, width=320, height=320"

INFER_OD()
{
  if [[ "$1" == "" || "$(($1%2))" == "0" ]]
  then
    echo "tiovxmultiscaler name=split$1"
    echo "src_0::roi-startx=360 src_0::roi-starty=180"
    echo "src_0::roi-width=1280 src_0::roi-height=720"
    echo "src_1::roi-startx=360 src_1::roi-starty=180"
    echo "src_1::roi-width=1280 src_1::roi-height=720"
    echo "src_2::roi-startx=360 src_2::roi-starty=180"
    echo "src_2::roi-width=1280 src_2::roi-height=720"
    echo "src_3::roi-startx=360 src_3::roi-starty=180"
    echo "src_3::roi-width=1280 src_3::roi-height=720"
    split_name="split$1"
  else
    split_name="split$(($1 - 1))"
  fi
  echo "$split_name. ! queue ! $MODEL_OD_CAPS !"
  echo "tiovxdlpreproc $MODEL_OD_PRE_PROC_PROPS !"
  echo "tidlinferer model=$MODEL_OD !"
  echo "post$1.tensor"
  echo "$split_name. ! queue ! $POST_PROC_CAPS !"
  echo "post$1.sink"
  echo "tidlpostproc $POST_PROC_PROPS name=post$1 model=$MODEL_OD"
}

MODEL_CL=/opt/model_zoo/TFL-CL-0000-mobileNetV1-mlperf
MODEL_CL_PRE_PROC_PROPS="data-type=3 channel-order=1 tensor-format=rgb out-pool-size=4"
MODEL_CL_CAPS="video/x-raw, width=224, height=224"

INFER_CL()
{
  if [[ "$1" == "" || "$(($1%2))" == "0" ]]
  then
    echo "tiovxmultiscaler name=split$1"
    echo "src_0::roi-startx=0 src_0::roi-starty=0"
    echo "src_0::roi-width=896 src_0::roi-height=896"
    echo "src_2::roi-startx=0 src_2::roi-starty=0"
    echo "src_2::roi-width=896 src_2::roi-height=896"
    split_name="split$1"
  else
    split_name="split$(($1 - 1))"
  fi
  echo "$split_name. ! queue ! $MODEL_CL_CAPS !"
  echo "tiovxdlpreproc $MODEL_CL_PRE_PROC_PROPS !"
  echo "tidlinferer model=$MODEL_CL !"
  echo "post$1.tensor"
  echo "$split_name. ! queue ! $POST_PROC_CAPS !"
  echo "post$1.sink"
  echo "tidlpostproc $POST_PROC_PROPS name=post$1 model=$MODEL_CL"
}

MODEL_SS=/opt/model_zoo/ONR-SS-8610-deeplabv3lite-mobv2-ade20k32-512x512
MODEL_SS_PRE_PROC_PROPS="data-type=3 channel-order=0 tensor-format=rgb out-pool-size=4"
MODEL_SS_CAPS="video/x-raw, width=512, height=512"

INFER_SS()
{
  if [[ "$1" == "" || "$(($1%2))" == "0" ]]
  then
    echo "tiovxmultiscaler name=split$1"
    split_name="split$1"
  else
    split_name="split$(($1 - 1))"
  fi
  echo "$split_name. ! queue ! $MODEL_SS_CAPS !"
  echo "tiovxdlpreproc $MODEL_SS_PRE_PROC_PROPS !"
  echo "tidlinferer model=$MODEL_SS !"
  echo "post$1.tensor"
  echo "$split_name. ! queue ! $POST_PROC_CAPS !"
  echo "post$1.sink"
  echo "tidlpostproc $POST_PROC_PROPS name=post$1 model=$MODEL_SS"
}
################################################################################

WINDOW_WIDTH=640
WINDOW_HEIGHT=360
NUM_WINDOWS_X=3
NUM_WINDOWS_Y=3
OUT_WIDTH=1920
OUT_HEIGHT=1080

MOSAIC()
{
  echo "tiovxmosaic name=mosaic target=1"
  for ((i=0;i<$1;i++))
  do
    startx=$(($i % $NUM_WINDOWS_X * $WINDOW_WIDTH));
    starty=$(($i/$NUM_WINDOWS_X % $NUM_WINDOWS_Y * $WINDOW_HEIGHT));
    echo "sink_$i::startx=<$startx> sink_$i::starty=<$starty>"
    echo "sink_$i::widths=<$WINDOW_WIDTH> sink_$i::heights=<$WINDOW_HEIGHT>"
  done
  echo "! video/x-raw, width=$OUT_WIDTH, height=$OUT_HEIGHT"
}

################################################################################

DISPLAY="kmssink sync=false driver-name=tidss"
ENCODE_H264="v4l2h264enc bitrate=10000000 ! fakesink sync=true"
ENCODE_H265="v4l2h265enc ! fakesink sync=true"

###############################################################################

GST_LAUNCH()
{
  sleep 2
  set -x
  gst-launch-1.0 $1
  set +x
}

################################################################################
############################## SISO TEST CASES #################################
################################################################################
SISO_TEST_CASE_0001()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0001"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (classification) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_CL) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0002()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0002"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (detection) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_OD) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0003()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0003"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (segmentation) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_SS) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0004()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0004"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (classification) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_CL) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0005()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0005"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (detection) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_OD) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0006()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0006"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (segmentation) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_SS) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SISO_TEST_CASE_0007()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0007"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (classification) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_CL) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SISO_TEST_CASE_0008()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0008"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (detection) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_OD) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SISO_TEST_CASE_0009()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0009"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 1x DLInferer (segmentation) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_SS) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0010()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0010"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 1x DLInferer (classification) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! $(INFER_CL) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0011()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0011"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 1x DLInferer (detection) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! $(INFER_OD) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0012()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0012"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 1x DLInferer (segmentation) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! $(INFER_SS) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0013()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0013"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 1x DLInferer (classification) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! $(INFER_CL) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0014()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0014"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 1x DLInferer (detection) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! $(INFER_OD) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0015()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0015"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 1x DLInferer (segmentation) - PostProc (1MP) - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! $(INFER_SS) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0016()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0016"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 1x DLInferer (classification) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! $(INFER_CL) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0017()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0017"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 1x DLInferer (detection) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! $(INFER_OD) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SISO_TEST_CASE_0018()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0018"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 1x DLInferer (segmentation) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! $(INFER_SS) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SISO_TEST_CASE_0019()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0019"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 1x DLInferer (classification) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! $(INFER_CL) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SISO_TEST_CASE_0020()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0020"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 1x DLInferer (detection) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! $(INFER_OD) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SISO_TEST_CASE_0021()
{
  echo "" >> $LOG_FILE

  NAME="SISO_TEST_CASE_0021"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 1x DLInferer (segmentation) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! $(INFER_SS) ! $(MOSAIC 1) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
############################## SIMO TEST CASES #################################
################################################################################
SIMO_TEST_CASE_0001()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0001"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (classification) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_CL 0) ! mosaic. \
                                   $(INFER_CL 1) ! mosaic. \
              src_split. ! queue ! $(INFER_CL 2) ! mosaic. \
                                   $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0002()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0002"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (detection) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_OD 0) ! mosaic. \
                                   $(INFER_OD 1) ! mosaic. \
              src_split. ! queue ! $(INFER_OD 2) ! mosaic. \
                                   $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0003()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0003"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (segmentation) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_SS 0) ! mosaic. \
                                   $(INFER_SS 1) ! mosaic. \
              src_split. ! queue ! $(INFER_SS 2) ! mosaic. \
                                   $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0004()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0004"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (classification) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_CL 0) ! mosaic. \
                                   $(INFER_CL 1) ! mosaic. \
              src_split. ! queue ! $(INFER_CL 2) ! mosaic. \
                                   $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0005()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0005"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (detection) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_OD 0) ! mosaic. \
                                   $(INFER_OD 1) ! mosaic. \
              src_split. ! queue ! $(INFER_OD 2) ! mosaic. \
                                   $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0006()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0006"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (segmentation) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_SS 0) ! mosaic. \
                                   $(INFER_SS 1) ! mosaic. \
              src_split. ! queue ! $(INFER_SS 2) ! mosaic. \
                                   $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SIMO_TEST_CASE_0007()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0007"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (classification) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_CL 0) ! mosaic. \
                                   $(INFER_CL 1) ! mosaic. \
              src_split. ! queue ! $(INFER_CL 2) ! mosaic. \
                                   $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SIMO_TEST_CASE_0008()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0008"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (detection) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_OD 0) ! mosaic. \
                                   $(INFER_OD 1) ! mosaic. \
              src_split. ! queue ! $(INFER_OD 2) ! mosaic. \
                                   $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SIMO_TEST_CASE_0009()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0009"
  TITLE="1x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (segmentation) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! tee name=src_split \
              src_split. ! queue ! $(INFER_SS 0) ! mosaic. \
                                   $(INFER_SS 1) ! mosaic. \
              src_split. ! queue ! $(INFER_SS 2) ! mosaic. \
                                   $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0010()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0010"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (classification) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_CL 0) ! mosaic. \
                                   $(INFER_CL 1) ! mosaic. \
              src_split. ! queue ! $(INFER_CL 2) ! mosaic. \
                                   $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0011()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0011"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (detection) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_OD 0) ! mosaic. \
                                   $(INFER_OD 1) ! mosaic. \
              src_split. ! queue ! $(INFER_OD 2) ! mosaic. \
                                   $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0012()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0012"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (segmentation) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_SS 0) ! mosaic. \
                                   $(INFER_SS 1) ! mosaic. \
              src_split. ! queue ! $(INFER_SS 2) ! mosaic. \
                                   $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0013()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0013"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (classification) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_CL 0) ! mosaic. \
                                   $(INFER_CL 1) ! mosaic. \
              src_split. ! queue ! $(INFER_CL 2) ! mosaic. \
                                   $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0014()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0014"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (detection) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_OD 0) ! mosaic. \
                                   $(INFER_OD 1) ! mosaic. \
              src_split. ! queue ! $(INFER_OD 2) ! mosaic. \
                                   $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0015()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0015"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (se) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_SS 0) ! mosaic. \
                                   $(INFER_SS 1) ! mosaic. \
              src_split. ! queue ! $(INFER_SS 2) ! mosaic. \
                                   $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0016()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0016"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (classification) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_CL 0) ! mosaic. \
                                   $(INFER_CL 1) ! mosaic. \
              src_split. ! queue ! $(INFER_CL 2) ! mosaic. \
                                   $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0017()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0017"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (detection) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_OD 0) ! mosaic. \
                                   $(INFER_OD 1) ! mosaic. \
              src_split. ! queue ! $(INFER_OD 2) ! mosaic. \
                                   $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
SIMO_TEST_CASE_0018()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0018"
  TITLE="1x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (segmentation) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_SS 0) ! mosaic. \
                                   $(INFER_SS 1) ! mosaic. \
              src_split. ! queue ! $(INFER_SS 2) ! mosaic. \
                                   $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SIMO_TEST_CASE_0019()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0019"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (classification) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_CL 0) ! mosaic. \
                                   $(INFER_CL 1) ! mosaic. \
              src_split. ! queue ! $(INFER_CL 2) ! mosaic. \
                                   $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SIMO_TEST_CASE_0020()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0020"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (detection) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_OD 0) ! mosaic. \
                                   $(INFER_OD 1) ! mosaic. \
              src_split. ! queue ! $(INFER_OD 2) ! mosaic. \
                                   $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

SIMO_TEST_CASE_0021()
{
  echo "" >> $LOG_FILE

  NAME="SIMO_TEST_CASE_0021"
  TITLE="1x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (segmentation) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP) ! tee name=src_split \
              src_split. ! queue ! $(INFER_SS 0) ! mosaic. \
                                   $(INFER_SS 1) ! mosaic. \
              src_split. ! queue ! $(INFER_SS 2) ! mosaic. \
                                   $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
############################## MIMO TEST CASES #################################
################################################################################
MIMO_TEST_CASE_0001()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0001"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (classification) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_CL 0) ! mosaic. \
                          $(INFER_CL 1) ! mosaic. \
              $IMX219_1 ! $(INFER_CL 2) ! mosaic. \
                          $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0002()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0002"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (detection) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_OD 0) ! mosaic. \
                          $(INFER_OD 1) ! mosaic. \
              $IMX219_1 ! $(INFER_OD 2) ! mosaic. \
                          $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0003()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0003"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (segmentation) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_SS 0) ! mosaic. \
                          $(INFER_SS 1) ! mosaic. \
              $IMX219_1 ! $(INFER_SS 2) ! mosaic. \
                          $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0004()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0004"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (classification) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_CL 0) ! mosaic. \
                          $(INFER_CL 1) ! mosaic. \
              $IMX219_1 ! $(INFER_CL 2) ! mosaic. \
                          $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0005()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0005"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (detection) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_OD 0) ! mosaic. \
                          $(INFER_OD 1) ! mosaic. \
              $IMX219_1 ! $(INFER_OD 2) ! mosaic. \
                          $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0006()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0006"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (segmentation) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_SS 0) ! mosaic. \
                          $(INFER_SS 1) ! mosaic. \
              $IMX219_1 ! $(INFER_SS 2) ! mosaic. \
                          $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

MIMO_TEST_CASE_0007()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0007"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (classification) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_CL 0) ! mosaic. \
                          $(INFER_CL 1) ! mosaic. \
              $IMX219_1 ! $(INFER_CL 2) ! mosaic. \
                          $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

MIMO_TEST_CASE_0008()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0008"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (detection) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_OD 0) ! mosaic. \
                          $(INFER_OD 1) ! mosaic. \
              $IMX219_1 ! $(INFER_OD 2) ! mosaic. \
                          $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

MIMO_TEST_CASE_0009()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0009"
  TITLE="2x IMX219 2MP @30fps - ISP - MSC - PreProc - 4x DLInferer (segmentation) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$IMX219_0 ! $(INFER_SS 0) ! mosaic. \
                          $(INFER_SS 1) ! mosaic. \
              $IMX219_1 ! $(INFER_SS 2) ! mosaic. \
                          $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0010()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0010"
  TITLE="2x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (classification) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP 0) ! $(INFER_CL 0) ! mosaic. \
                                    $(INFER_CL 1) ! mosaic. \
              $(VIDEO_H264_2MP 1) ! $(INFER_CL 2) ! mosaic. \
                                    $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0011()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0011"
  TITLE="2x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (detection) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP 0) ! $(INFER_OD 0) ! mosaic. \
                                    $(INFER_OD 1) ! mosaic. \
              $(VIDEO_H264_2MP 1) ! $(INFER_OD 2) ! mosaic. \
                                    $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0012()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0012"
  TITLE="2x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (segmentation) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP 0) ! $(INFER_SS 0) ! mosaic. \
                                    $(INFER_SS 1) ! mosaic. \
              $(VIDEO_H264_2MP 1) ! $(INFER_SS 2) ! mosaic. \
                                    $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0013()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0013"
  TITLE="2x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (classification) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP 0) ! $(INFER_CL 0) ! mosaic. \
                                    $(INFER_CL 1) ! mosaic. \
              $(VIDEO_H265_2MP 1) ! $(INFER_CL 2) ! mosaic. \
                                    $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0014()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0014"
  TITLE="2x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (detection) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP 0) ! $(INFER_OD 0) ! mosaic. \
                                    $(INFER_OD 1) ! mosaic. \
              $(VIDEO_H265_2MP 1) ! $(INFER_OD 2) ! mosaic. \
                                    $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0015()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0015"
  TITLE="2x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (se) - PostProc - Mosaic (2MP) - Display"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP 0) ! $(INFER_SS 0) ! mosaic. \
                                    $(INFER_SS 1) ! mosaic. \
              $(VIDEO_H265_2MP 1) ! $(INFER_SS 2) ! mosaic. \
                                    $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $DISPLAY"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0016()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0016"
  TITLE="2x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (classification) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP 0) ! $(INFER_CL 0) ! mosaic. \
                                    $(INFER_CL 1) ! mosaic. \
              $(VIDEO_H264_2MP 1) ! $(INFER_CL 2) ! mosaic. \
                                    $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0017()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0017"
  TITLE="2x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (detection) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP 0) ! $(INFER_OD 0) ! mosaic. \
                                    $(INFER_OD 1) ! mosaic. \
              $(VIDEO_H264_2MP 1) ! $(INFER_OD 2) ! mosaic. \
                                    $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################
MIMO_TEST_CASE_0018()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0018"
  TITLE="2x video 2MP @30fps - H.264 Decode - MSC - PreProc - 4x DLInferer (segmentation) - PostProc (2MP) - H.264 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H264_2MP 0) ! $(INFER_SS 0) ! mosaic. \
                                    $(INFER_SS 1) ! mosaic. \
              $(VIDEO_H264_2MP 1) ! $(INFER_SS 2) ! mosaic. \
                                    $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H264"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

MIMO_TEST_CASE_0019()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0019"
  TITLE="2x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (classification) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP 0) ! $(INFER_CL 0) ! mosaic. \
                                    $(INFER_CL 1) ! mosaic. \
              $(VIDEO_H265_2MP 1) ! $(INFER_CL 2) ! mosaic. \
                                    $(INFER_CL 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

MIMO_TEST_CASE_0020()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0020"
  TITLE="2x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (detection) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP 0) ! $(INFER_OD 0) ! mosaic. \
                                    $(INFER_OD 1) ! mosaic. \
              $(VIDEO_H265_2MP 1) ! $(INFER_OD 2) ! mosaic. \
                                    $(INFER_OD 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

MIMO_TEST_CASE_0021()
{
  echo "" >> $LOG_FILE

  NAME="MIMO_TEST_CASE_0021"
  TITLE="2x video 2MP @30fps - H.265 Decode - MSC - PreProc - 4x DLInferer (segmentation) - PostProc (2MP) - H.265 Encode (IPP | High | 2MP @ 30fps | 10Mbps)"
  echo $NAME
  echo "" >> $LOG_FILE

  GST_LAUNCH "$(VIDEO_H265_2MP 0) ! $(INFER_SS 0) ! mosaic. \
                                    $(INFER_SS 1) ! mosaic. \
              $(VIDEO_H265_2MP 1) ! $(INFER_SS 2) ! mosaic. \
                                    $(INFER_SS 3) ! mosaic. \
              $(MOSAIC 4) ! $PERF name=\"$NAME\" title=\"$TITLE\" ! $ENCODE_H265"

  if [ "$?" != "0" ]; then exit; fi
  echo "" >> $LOG_FILE
}

################################################################################

if [ "$IMX219_0_DEV" != "" ]
then
  SISO_TEST_CASE_0001
  SISO_TEST_CASE_0002
  SISO_TEST_CASE_0003
  SISO_TEST_CASE_0004
  SISO_TEST_CASE_0005
  SISO_TEST_CASE_0006
  if [ "$SOC" != "j721e" ]
  then
    SISO_TEST_CASE_0007
    SISO_TEST_CASE_0008
    SISO_TEST_CASE_0009
  fi
fi
SISO_TEST_CASE_0010
SISO_TEST_CASE_0011
SISO_TEST_CASE_0012
SISO_TEST_CASE_0013
SISO_TEST_CASE_0014
SISO_TEST_CASE_0015
SISO_TEST_CASE_0016
SISO_TEST_CASE_0017
SISO_TEST_CASE_0018
if [ "$SOC" != "j721e" ]
then
  SISO_TEST_CASE_0019
  SISO_TEST_CASE_0020
  SISO_TEST_CASE_0021
fi

if [ "$IMX219_0_DEV" != "" ]
then
  SIMO_TEST_CASE_0001
  SIMO_TEST_CASE_0002
  SIMO_TEST_CASE_0003
  SIMO_TEST_CASE_0004
  SIMO_TEST_CASE_0005
  SIMO_TEST_CASE_0006
  if [ "$SOC" != "j721e" ]
  then
    SIMO_TEST_CASE_0007
    SIMO_TEST_CASE_0008
    SIMO_TEST_CASE_0009
  fi
fi
SIMO_TEST_CASE_0010
SIMO_TEST_CASE_0011
SIMO_TEST_CASE_0012
SIMO_TEST_CASE_0013
SIMO_TEST_CASE_0014
SIMO_TEST_CASE_0015
SIMO_TEST_CASE_0016
SIMO_TEST_CASE_0017
SIMO_TEST_CASE_0018
if [ "$SOC" != "j721e" ]
then
  SIMO_TEST_CASE_0019
  SIMO_TEST_CASE_0020
  SIMO_TEST_CASE_0021
fi

if [[ "$IMX219_0_DEV" != "" && "$IMX219_1_DEV" != "" ]]
then
  MIMO_TEST_CASE_0001
  MIMO_TEST_CASE_0002
  MIMO_TEST_CASE_0003
  MIMO_TEST_CASE_0004
  MIMO_TEST_CASE_0005
  MIMO_TEST_CASE_0006
  if [ "$SOC" != "j721e" ]
  then
    MIMO_TEST_CASE_0007
    MIMO_TEST_CASE_0008
    MIMO_TEST_CASE_0009
  fi
fi
MIMO_TEST_CASE_0010
MIMO_TEST_CASE_0011
MIMO_TEST_CASE_0012
MIMO_TEST_CASE_0013
MIMO_TEST_CASE_0014
MIMO_TEST_CASE_0015
MIMO_TEST_CASE_0016
MIMO_TEST_CASE_0017
MIMO_TEST_CASE_0018
if [ "$SOC" != "j721e" ]
then
  MIMO_TEST_CASE_0019
  MIMO_TEST_CASE_0020
  MIMO_TEST_CASE_0021
fi