#!/bin/bash
#
# Start Real-time RL Controller for Production
#
# Usage:
#   ./start_controller.sh [options]
#
# Examples:
#   ./start_controller.sh --agent qmix --model checkpoints/qmix_best.pt
#   ./start_controller.sh --agent sac-gru --servers "192.168.1.10-13"
#

set -e

# Default configuration
AGENT_TYPE="qmix"
NUM_SERVERS=16
NUM_AGENTS=4
MODEL_PATH=""
SHM_PATH="/dev/shm/lb_rl_shm"
UPDATE_INTERVAL=0.2
HEALTH_CHECK_INTERVAL=5.0
PROMETHEUS_PORT=9090
MIN_HEALTHY_SERVERS=1
SERVER_IPS=""
LOG_FILE="realtime_controller.log"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --agent TYPE              Agent type: qmix or sac-gru (default: qmix)"
    echo "  --num-servers N           Number of servers (default: 16)"
    echo "  --num-agents N            Number of agents for QMIX (default: 4)"
    echo "  --model PATH              Path to pretrained model (required)"
    echo "  --shm-path PATH           Shared memory path (default: /dev/shm/lb_rl_shm)"
    echo "  --update-interval SEC     Update interval in seconds (default: 0.2)"
    echo "  --health-check SEC        Health check interval (default: 5.0)"
    echo "  --prometheus-port PORT    Prometheus port (default: 9090)"
    echo "  --min-servers N           Minimum healthy servers (default: 1)"
    echo "  --servers IPS             Space-separated server IPs"
    echo "  --log-file FILE           Log file path (default: realtime_controller.log)"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Example:"
    echo "  $0 --agent qmix --model ../problem-06-vpp-integration/checkpoints/qmix_best.pt \\"
    echo "     --servers \"192.168.1.10 192.168.1.11 192.168.1.12 192.168.1.13\""
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agent)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --num-servers)
            NUM_SERVERS="$2"
            shift 2
            ;;
        --num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --shm-path)
            SHM_PATH="$2"
            shift 2
            ;;
        --update-interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --health-check)
            HEALTH_CHECK_INTERVAL="$2"
            shift 2
            ;;
        --prometheus-port)
            PROMETHEUS_PORT="$2"
            shift 2
            ;;
        --min-servers)
            MIN_HEALTHY_SERVERS="$2"
            shift 2
            ;;
        --servers)
            SERVER_IPS="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: --model is required${NC}"
    usage
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

# Check if shared memory exists
if [ ! -e "$SHM_PATH" ]; then
    echo -e "${YELLOW}Warning: Shared memory not found: $SHM_PATH${NC}"
    echo -e "${YELLOW}Make sure VPP is running with 'lb rl enable'${NC}"
fi

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}Warning: Conda environment not active${NC}"
    echo -e "${YELLOW}Activating 'marl' environment...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate marl
fi

# Print configuration
echo -e "${GREEN}=== Real-time RL Controller ===${NC}"
echo "Agent Type:        $AGENT_TYPE"
echo "Num Servers:       $NUM_SERVERS"
echo "Num Agents:        $NUM_AGENTS"
echo "Model Path:        $MODEL_PATH"
echo "SHM Path:          $SHM_PATH"
echo "Update Interval:   ${UPDATE_INTERVAL}s"
echo "Health Check:      ${HEALTH_CHECK_INTERVAL}s"
echo "Prometheus Port:   $PROMETHEUS_PORT"
echo "Min Servers:       $MIN_HEALTHY_SERVERS"
if [ -n "$SERVER_IPS" ]; then
    echo "Server IPs:        $SERVER_IPS"
fi
echo "Log File:          $LOG_FILE"
echo ""

# Build command
CMD="python src/realtime_controller.py"
CMD="$CMD --agent-type $AGENT_TYPE"
CMD="$CMD --num-servers $NUM_SERVERS"
CMD="$CMD --num-agents $NUM_AGENTS"
CMD="$CMD --model-path $MODEL_PATH"
CMD="$CMD --shm-path $SHM_PATH"
CMD="$CMD --update-interval $UPDATE_INTERVAL"
CMD="$CMD --health-check-interval $HEALTH_CHECK_INTERVAL"
CMD="$CMD --prometheus-port $PROMETHEUS_PORT"
CMD="$CMD --min-healthy-servers $MIN_HEALTHY_SERVERS"

if [ -n "$SERVER_IPS" ]; then
    CMD="$CMD --server-ips $SERVER_IPS"
fi

# Check if controller is already running
if pgrep -f "realtime_controller.py" > /dev/null; then
    echo -e "${RED}Error: Controller is already running${NC}"
    echo "Stop it first with: ./scripts/stop_all.sh"
    exit 1
fi

# Start controller
echo -e "${GREEN}Starting controller...${NC}"
echo "Command: $CMD"
echo ""

# Run in background with logging
nohup $CMD > "$LOG_FILE" 2>&1 &
PID=$!

# Wait for startup
sleep 2

# Check if process is running
if ps -p $PID > /dev/null; then
    echo -e "${GREEN}✓ Controller started successfully (PID: $PID)${NC}"
    echo ""
    echo "Monitor logs:      tail -f $LOG_FILE"
    echo "Check metrics:     curl http://localhost:$PROMETHEUS_PORT/metrics"
    echo "Stop controller:   kill $PID  OR  ./scripts/stop_all.sh"
    echo ""
    
    # Save PID
    echo $PID > .controller.pid
else
    echo -e "${RED}✗ Controller failed to start${NC}"
    echo "Check logs: cat $LOG_FILE"
    exit 1
fi
