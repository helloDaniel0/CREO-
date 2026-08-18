#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/tcp-leocc.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

namespace ns3
{

class HandoverFifoQueueDisc : public QueueDisc
{
  public:
    static TypeId GetTypeId();
    HandoverFifoQueueDisc();
    void SetGateOpen(bool open);

  private:
    bool DoEnqueue(Ptr<QueueDiscItem> item) override;
    Ptr<QueueDiscItem> DoDequeue() override;
    Ptr<const QueueDiscItem> DoPeek() override;
    bool CheckConfig() override;
    void InitializeParams() override;

    bool m_gateOpen{true};
};

NS_OBJECT_ENSURE_REGISTERED(HandoverFifoQueueDisc);

TypeId
HandoverFifoQueueDisc::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::HandoverFifoQueueDisc")
            .SetParent<QueueDisc>()
            .SetGroupName("TrafficControl")
            .AddConstructor<HandoverFifoQueueDisc>()
            .AddAttribute("MaxSize",
                          "The maximum queue size",
                          QueueSizeValue(QueueSize("1000p")),
                          MakeQueueSizeAccessor(&QueueDisc::SetMaxSize, &QueueDisc::GetMaxSize),
                          MakeQueueSizeChecker());
    return tid;
}

HandoverFifoQueueDisc::HandoverFifoQueueDisc()
    : QueueDisc(QueueDiscSizePolicy::SINGLE_INTERNAL_QUEUE)
{
}

void
HandoverFifoQueueDisc::SetGateOpen(bool open)
{
    if (m_gateOpen == open)
    {
        return;
    }
    m_gateOpen = open;
    if (m_gateOpen)
    {
        Run();
    }
}

bool
HandoverFifoQueueDisc::DoEnqueue(Ptr<QueueDiscItem> item)
{
    if (GetCurrentSize() + item > GetMaxSize())
    {
        DropBeforeEnqueue(item, "Handover FIFO queue limit exceeded");
        return false;
    }
    return GetInternalQueue(0)->Enqueue(item);
}

Ptr<QueueDiscItem>
HandoverFifoQueueDisc::DoDequeue()
{
    return m_gateOpen ? GetInternalQueue(0)->Dequeue() : nullptr;
}

Ptr<const QueueDiscItem>
HandoverFifoQueueDisc::DoPeek()
{
    return m_gateOpen ? GetInternalQueue(0)->Peek() : nullptr;
}

bool
HandoverFifoQueueDisc::CheckConfig()
{
    if (GetNQueueDiscClasses() > 0 || GetNPacketFilters() > 0)
    {
        return false;
    }
    if (GetNInternalQueues() == 0)
    {
        AddInternalQueue(
            CreateObjectWithAttributes<DropTailQueue<QueueDiscItem>>("MaxSize",
                                                                     QueueSizeValue(GetMaxSize())));
    }
    return GetNInternalQueues() == 1;
}

void
HandoverFifoQueueDisc::InitializeParams()
{
}

} // namespace ns3

namespace
{

struct TraceSample
{
    double firstRateMbps;
    double secondRateMbps;
    double firstDelayMs;
    double secondDelayMs;
};

std::vector<TraceSample> g_trace;
NetDeviceContainer g_dishSatellite;
NetDeviceContainer g_satelliteGs;
Ptr<PacketSink> g_sink;
Ptr<FlowMonitor> g_monitor;
Ptr<QueueDisc> g_firstQueue;
Ptr<QueueDisc> g_secondQueue;
Ptr<Queue<Packet>> g_firstDeviceQueue;
Ptr<Queue<Packet>> g_secondDeviceQueue;
Ptr<RateErrorModel> g_handoverForwardErrorModel;
Ptr<RateErrorModel> g_handoverReverseErrorModel;
Ptr<HandoverFifoQueueDisc> g_handoverQueue;

std::ofstream g_throughput;
std::ofstream g_txThroughput;
std::ofstream g_realBw;
std::ofstream g_rtt;
std::ofstream g_prop;
std::ofstream g_queue;
std::ofstream g_cwnd;
std::ofstream g_pacing;
std::ofstream g_leoccModel;
std::ofstream g_leoccMode;
std::ofstream g_loss;
std::ofstream g_linkRates;
std::ofstream g_handover;

double g_samplePeriod = 1.0;
double g_linkUpdatePeriod = 0.5;
double g_fixedDelayMs = 1.0;
double g_throughputPeriod = 0.5;
double g_diagnosticPeriod = 0.1;
double g_bandwidthJitterStd = 0.10;
uint32_t g_segmentSize = 1000;
uint64_t g_previousRxBytes = 0;
Time g_previousMeasurement = Seconds(0);
uint64_t g_previousTxBytes = 0;
Time g_previousTxMeasurement = Seconds(0);
double g_currentFirstRateMbps = 0.0;
double g_currentSecondRateMbps = 0.0;
double g_currentBottleneckMbps = 0.0;
double g_handoverDurationMs = 50.0;
double g_backgroundErrorRate = 0.0;
bool g_handoverActive = false;
bool g_traceLeoCcModel = true;
std::mt19937 g_jitterGenerator(1);

std::vector<std::vector<double>>
ReadNumericRows(const std::string& path)
{
    std::ifstream input(path);
    NS_ABORT_MSG_IF(!input.is_open(), "Cannot open trace file: " << path);

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream stream(line);
        std::vector<double> row;
        double value;
        while (stream >> value)
        {
            row.push_back(value);
        }
        if (!row.empty())
        {
            rows.push_back(row);
        }
    }
    return rows;
}

void
LoadTrace(const std::string& bandwidthPath,
          const std::string& latencyPath,
          double delayScale)
{
    const auto bandwidth = ReadNumericRows(bandwidthPath);
    const auto latency = ReadNumericRows(latencyPath);
    const std::size_t count = std::min(bandwidth.size(), latency.size());
    NS_ABORT_MSG_IF(count == 0, "The bandwidth/latency trace pair is empty");

    g_trace.clear();
    g_trace.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
    {
        NS_ABORT_MSG_IF(bandwidth[index].size() < 3 || latency[index].size() < 3,
                        "Trace rows must contain time and two values");
        g_trace.push_back({bandwidth[index][1],
                           bandwidth[index][2],
                           latency[index][1] * delayScale,
                           latency[index][2] * delayScale});
    }
}

double
MinimumDynamicOneWayDelay()
{
    double minimum = g_trace.front().firstDelayMs + g_trace.front().secondDelayMs;
    for (const auto& sample : g_trace)
    {
        minimum = std::min(minimum, sample.firstDelayMs + sample.secondDelayMs);
    }
    return minimum;
}

void
UpdateLinks()
{
    const Time nowTime = Simulator::Now();
    const double now = nowTime.GetSeconds();
    const int64_t samplePeriodNs = Seconds(g_samplePeriod).GetNanoSeconds();
    const auto index = static_cast<std::size_t>(nowTime.GetNanoSeconds() / samplePeriodNs) %
                       g_trace.size();
    const auto& sample = g_trace[index];

    // Match the original Fig. 10/11 scripts: each link receives an independent
    // multiplicative N(0, 0.10) perturbation whenever its rate is written.
    std::normal_distribution<double> jitter(0.0, g_bandwidthJitterStd);
    const double firstRateMbps =
        std::max(0.01,
                 std::round(sample.firstRateMbps * (1.0 + jitter(g_jitterGenerator)) * 100.0) /
                     100.0);
    const double secondRateMbps =
        std::max(0.01,
                 std::round(sample.secondRateMbps * (1.0 + jitter(g_jitterGenerator)) * 100.0) /
                     100.0);

    auto firstDevice = DynamicCast<PointToPointNetDevice>(g_dishSatellite.Get(0));
    auto secondDevice = DynamicCast<PointToPointNetDevice>(g_satelliteGs.Get(0));
    firstDevice->SetAttribute("DataRate",
                              DataRateValue(DataRate(static_cast<uint64_t>(firstRateMbps * 1e6))));
    secondDevice->SetAttribute("DataRate",
                               DataRateValue(DataRate(static_cast<uint64_t>(secondRateMbps * 1e6))));

    auto firstChannel = DynamicCast<PointToPointChannel>(firstDevice->GetChannel());
    auto secondChannel = DynamicCast<PointToPointChannel>(secondDevice->GetChannel());
    firstChannel->SetAttribute("Delay", TimeValue(MilliSeconds(sample.firstDelayMs)));
    secondChannel->SetAttribute("Delay", TimeValue(MilliSeconds(sample.secondDelayMs)));

    g_currentFirstRateMbps = firstRateMbps;
    g_currentSecondRateMbps = secondRateMbps;
    g_currentBottleneckMbps = std::min(firstRateMbps, secondRateMbps);
    const double effectiveBottleneck = g_handoverActive ? 0.0 : g_currentBottleneckMbps;
    const double baseRtt =
        2.0 * (sample.firstDelayMs + sample.secondDelayMs + g_fixedDelayMs);
    g_realBw << now << " " << effectiveBottleneck << std::endl;
    g_prop << now << " " << baseRtt << std::endl;
    g_linkRates << now << " " << (g_handoverActive ? 0.0 : firstRateMbps) << " "
                << (g_handoverActive ? 0.0 : secondRateMbps) << std::endl;

    Simulator::Schedule(Seconds(g_linkUpdatePeriod), &UpdateLinks);
}

uint64_t
GetForwardTxBytes()
{
    uint64_t selectedBytes = 0;
    for (const auto& [flowId, stats] : g_monitor->GetFlowStats())
    {
        (void)flowId;
        selectedBytes = std::max(selectedBytes, stats.txBytes);
    }
    return selectedBytes;
}

void
TraceTxThroughput()
{
    const Time now = Simulator::Now();
    const uint64_t transmitted = g_monitor ? GetForwardTxBytes() : 0;
    const double interval = (now - g_previousTxMeasurement).GetSeconds();
    const double throughputMbps = interval > 0
                                      ? (transmitted - g_previousTxBytes) * 8.0 / interval / 1e6
                                      : 0.0;
    g_txThroughput << now.GetSeconds() << " " << throughputMbps << std::endl;
    g_previousTxBytes = transmitted;
    g_previousTxMeasurement = now;

    Simulator::Schedule(Seconds(g_throughputPeriod), &TraceTxThroughput);
}

void
DropOldSatellitePackets()
{
    if (g_secondQueue && g_secondQueue->GetNInternalQueues() > 0)
    {
        g_secondQueue->GetInternalQueue(0)->Flush();
    }
    if (g_secondDeviceQueue)
    {
        g_secondDeviceQueue->Flush();
    }
}

void
EndHandover()
{
    // Packets still buffered on the old satellite cannot migrate to the new path.
    DropOldSatellitePackets();
    g_handoverActive = false;
    g_handoverForwardErrorModel->SetAttribute("ErrorRate",
                                               DoubleValue(g_backgroundErrorRate));
    g_handoverReverseErrorModel->SetAttribute("ErrorRate", DoubleValue(0.0));
    g_handoverQueue->SetGateOpen(true);
    const double now = Simulator::Now().GetSeconds();
    g_realBw << now << " " << g_currentBottleneckMbps << std::endl;
    g_linkRates << now << " " << g_currentFirstRateMbps << " " << g_currentSecondRateMbps
                << std::endl;
    g_handover << now << " end" << std::endl;
}

void
BeginHandover()
{
    g_handoverActive = true;
    g_handoverQueue->SetGateOpen(false);
    g_handoverForwardErrorModel->SetAttribute("ErrorRate", DoubleValue(1.0));
    g_handoverReverseErrorModel->SetAttribute("ErrorRate", DoubleValue(1.0));
    DropOldSatellitePackets();

    const double now = Simulator::Now().GetSeconds();
    g_realBw << now << " 0" << std::endl;
    g_linkRates << now << " 0 0" << std::endl;
    g_handover << now << " start" << std::endl;
    Simulator::Schedule(MilliSeconds(g_handoverDurationMs), &EndHandover);
}

void
NotifyLeoCcReconfiguration()
{
    auto tcp = NodeList::GetNode(0)->GetObject<TcpL4Protocol>();
    NS_ABORT_MSG_IF(!tcp, "The sender has no TcpL4Protocol");
    bool notified = false;
    for (const auto& [socketId, socket] : tcp->GetSockets())
    {
        (void)socketId;
        PointerValue congestionValue;
        socket->GetAttribute("CongestionOps", congestionValue);
        auto leocc = DynamicCast<TcpLeoCC>(congestionValue.Get<TcpCongestionOps>());
        if (leocc)
        {
            leocc->NotifyReconfiguration();
            notified = true;
        }
    }
    NS_ABORT_MSG_IF(!notified, "RI detector could not find a TcpLeoCC sender socket");
    g_handover << Simulator::Now().GetSeconds() << " ri-detected" << std::endl;
}

void
TraceThroughput()
{
    const Time now = Simulator::Now();
    const uint64_t received = g_sink ? g_sink->GetTotalRx() : 0;
    const double interval = (now - g_previousMeasurement).GetSeconds();
    const double throughputMbps = interval > 0
                                      ? (received - g_previousRxBytes) * 8.0 / interval / 1e6
                                      : 0.0;
    g_throughput << now.GetSeconds() << " " << throughputMbps << std::endl;
    g_previousRxBytes = received;
    g_previousMeasurement = now;

    Simulator::Schedule(Seconds(g_throughputPeriod), &TraceThroughput);
}

void
TraceDiagnostics()
{
    const Time now = Simulator::Now();
    uint64_t txPackets = 0;
    uint64_t lostPackets = 0;
    for (const auto& [flowId, stats] : g_monitor->GetFlowStats())
    {
        (void)flowId;
        if (stats.txBytes > 0 && stats.txPackets >= txPackets)
        {
            txPackets = stats.txPackets;
            lostPackets = stats.lostPackets;
        }
    }
    const double lossRate = txPackets > 0 ? static_cast<double>(lostPackets) / txPackets : 0.0;
    g_loss << now.GetSeconds() << " " << lossRate << std::endl;

    const uint32_t firstDisc = g_firstQueue ? g_firstQueue->GetCurrentSize().GetValue() : 0;
    const uint32_t secondDisc = g_secondQueue ? g_secondQueue->GetCurrentSize().GetValue() : 0;
    const uint32_t firstDevice = g_firstDeviceQueue ? g_firstDeviceQueue->GetNPackets() : 0;
    const uint32_t secondDevice = g_secondDeviceQueue ? g_secondDeviceQueue->GetNPackets() : 0;
    const uint32_t first = firstDisc + firstDevice;
    const uint32_t second = secondDisc + secondDevice;
    g_queue << now.GetSeconds() << " " << std::max(first, second) << std::endl;

    Simulator::Schedule(Seconds(g_diagnosticPeriod), &TraceDiagnostics);
}

void
CwndTracer(uint32_t oldValue, uint32_t newValue)
{
    (void)oldValue;
    g_cwnd << Simulator::Now().GetSeconds() << " "
           << static_cast<double>(newValue) / g_segmentSize << std::endl;
}

void
RttTracer(Time oldValue, Time newValue)
{
    (void)oldValue;
    g_rtt << Simulator::Now().GetSeconds() << " " << newValue.GetMicroSeconds() / 1000.0
          << std::endl;
}

void
PacingTracer(DataRate oldValue, DataRate newValue)
{
    (void)oldValue;
    g_pacing << Simulator::Now().GetSeconds() << " " << newValue.GetBitRate() / 1e6
             << std::endl;
}

void
AggressiveRateTracer(DataRate oldValue, DataRate newValue)
{
    (void)oldValue;
    g_leoccModel << Simulator::Now().GetSeconds() << " aggressive "
                 << newValue.GetBitRate() / 1e6 << std::endl;
}

void
ModerateRateTracer(DataRate oldValue, DataRate newValue)
{
    (void)oldValue;
    g_leoccModel << Simulator::Now().GetSeconds() << " moderate "
                 << newValue.GetBitRate() / 1e6 << std::endl;
}

void
TargetRateTracer(DataRate oldValue, DataRate newValue)
{
    (void)oldValue;
    g_leoccModel << Simulator::Now().GetSeconds() << " target " << newValue.GetBitRate() / 1e6
                 << std::endl;
}

void
ModeTracer(uint32_t oldValue, uint32_t newValue)
{
    (void)oldValue;
    g_leoccMode << Simulator::Now().GetSeconds() << " " << newValue << std::endl;
}

void
ConnectTcpTraces()
{
    const std::string socketPath = "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/";
    Config::ConnectWithoutContext(socketPath + "CongestionWindow", MakeCallback(&CwndTracer));
    Config::ConnectWithoutContext(socketPath + "RTT", MakeCallback(&RttTracer));
    Config::ConnectWithoutContext(socketPath + "PacingRate", MakeCallback(&PacingTracer));
    if (!g_traceLeoCcModel)
    {
        return;
    }
    const std::string leoccPath = socketPath + "CongestionOps/$ns3::TcpLeoCC/";
    Config::ConnectWithoutContext(leoccPath + "AggressiveBw", MakeCallback(&AggressiveRateTracer));
    Config::ConnectWithoutContext(leoccPath + "ModerateBw", MakeCallback(&ModerateRateTracer));
    Config::ConnectWithoutContext(leoccPath + "TargetRate", MakeCallback(&TargetRateTracer));
    Config::ConnectWithoutContext(leoccPath + "Mode", MakeCallback(&ModeTracer));
}

void
OpenOutputs(const std::string& outputDirectory)
{
    std::filesystem::create_directories(outputDirectory);
    g_throughput.open(outputDirectory + "/throughput.dat");
    g_txThroughput.open(outputDirectory + "/tx-throughput.dat");
    g_realBw.open(outputDirectory + "/realbw.dat");
    g_rtt.open(outputDirectory + "/rtt.dat");
    g_prop.open(outputDirectory + "/prop.dat");
    g_queue.open(outputDirectory + "/queueSize.dat");
    g_cwnd.open(outputDirectory + "/cwnd.dat");
    g_pacing.open(outputDirectory + "/pacing.dat");
    g_leoccModel.open(outputDirectory + "/leocc_model.dat");
    g_leoccMode.open(outputDirectory + "/leocc_mode.dat");
    g_loss.open(outputDirectory + "/loss.dat");
    g_linkRates.open(outputDirectory + "/link_rates.dat");
    g_handover.open(outputDirectory + "/handover.dat");
    NS_ABORT_MSG_IF(!g_throughput || !g_txThroughput || !g_realBw || !g_rtt || !g_prop || !g_queue ||
                        !g_cwnd || !g_pacing || !g_leoccModel || !g_leoccMode || !g_loss ||
                        !g_linkRates || !g_handover,
                    "Failed to open one or more result files under " << outputDirectory);
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string traceSet = "generated";
    std::string pathMode = "BP";
    std::string bandwidthTrace;
    std::string latencyTrace;
    std::string outputDirectory;
    std::string tcpTypeId = "TcpLeoCC";
    double stopTime = 200.0;
    double targetMinRttMs = 0.0;
    double errorRate = 0.005;
    double reconfigurationInterval = 15.0;
    double reconfigurationOffset = 12.0;
    double handoverTime = 15.0;
    double reconfigurationNotificationTime = -1.0;
    bool enableReconfiguration = false;
    bool enableHandover = false;
    uint32_t jitterSeed = 1;
    uint32_t queuePackets = 2000;
    uint32_t deviceQueuePackets = 100;

    CommandLine command(__FILE__);
    command.AddValue("traceSet", "generated or sigcomm", traceSet);
    command.AddValue("pathMode", "BP or ISL", pathMode);
    command.AddValue("bwTrace", "Bandwidth trace path", bandwidthTrace);
    command.AddValue("latencyTrace", "Latency trace path", latencyTrace);
    command.AddValue("outputDir", "Result directory", outputDirectory);
    command.AddValue("tcpTypeId", "TCP TypeId without ns3:: prefix", tcpTypeId);
    command.AddValue("stopTime", "Simulation duration in seconds", stopTime);
    command.AddValue("throughputPeriod",
                     "Receiver throughput sampling period in seconds",
                     g_throughputPeriod);
    command.AddValue("diagnosticPeriod",
                     "Queue and loss sampling period in seconds",
                     g_diagnosticPeriod);
    command.AddValue("bandwidthJitterStd",
                     "Stddev of multiplicative Gaussian bandwidth jitter",
                     g_bandwidthJitterStd);
    command.AddValue("jitterSeed", "Seed of the deterministic bandwidth jitter", jitterSeed);
    command.AddValue("targetMinRtt",
                     "Target minimum RTT in ms; zero uses the generated-path default or "
                     "native SIGCOMM RTT",
                     targetMinRttMs);
    command.AddValue("errorRate", "Wireless packet error rate", errorRate);
    command.AddValue("queuePackets", "Fifo queue length in packets", queuePackets);
    command.AddValue("deviceQueuePackets",
                     "Point-to-point device queue length in packets",
                     deviceQueuePackets);
    command.AddValue("enableReconfiguration", "Enable LeoCC periodic reconfiguration adaptation", enableReconfiguration);
    command.AddValue("reconfigurationInterval", "LeoCC reconfiguration period in seconds", reconfigurationInterval);
    command.AddValue("reconfigurationOffset", "First LeoCC reconfiguration in seconds", reconfigurationOffset);
    command.AddValue("enableHandover", "Inject one satellite handover interruption", enableHandover);
    command.AddValue("handoverTime", "Handover start time in seconds", handoverTime);
    command.AddValue("handoverDurationMs", "Handover interruption duration in ms", g_handoverDurationMs);
    command.AddValue("reconfigurationNotificationTime",
                     "Absolute time of an external LeoCC RI detection event; negative disables it",
                     reconfigurationNotificationTime);
    command.Parse(argc, argv);

    NS_ABORT_MSG_IF(traceSet != "generated" && traceSet != "sigcomm",
                    "traceSet must be generated or sigcomm");
    NS_ABORT_MSG_IF(pathMode != "BP" && pathMode != "ISL", "pathMode must be BP or ISL");
    NS_ABORT_MSG_IF(g_throughputPeriod <= 0.0, "throughputPeriod must be positive");
    NS_ABORT_MSG_IF(g_bandwidthJitterStd < 0.0, "bandwidthJitterStd cannot be negative");
    NS_ABORT_MSG_IF(enableHandover && (handoverTime <= 0.1 || handoverTime >= stopTime),
                    "handoverTime must occur while the flow is active");
    NS_ABORT_MSG_IF(g_handoverDurationMs <= 0.0, "handoverDurationMs must be positive");
    g_jitterGenerator.seed(jitterSeed);
    g_traceLeoCcModel = tcpTypeId == "TcpLeoCC";

    if (bandwidthTrace.empty())
    {
        bandwidthTrace = traceSet == "generated" ? "dataset/bw.txt" : "dataset/SIGCOMMbw.txt";
    }
    if (latencyTrace.empty())
    {
        latencyTrace =
            traceSet == "generated" ? "dataset/latency.txt" : "dataset/SIGCOMMlatency.txt";
    }
    if (outputDirectory.empty())
    {
        outputDirectory = "results/connected/" + pathMode + "-LeoCC-" + traceSet;
    }
    g_samplePeriod = traceSet == "generated" ? 1.0 : 0.1;
    g_linkUpdatePeriod = traceSet == "generated" ? 0.5 : 0.1;
    if (traceSet == "generated")
    {
        // Generated delays are stored in seconds; normalize them to milliseconds.
        auto latencyRows = ReadNumericRows(latencyTrace);
        for (auto& row : latencyRows)
        {
            if (row.size() >= 3)
            {
                row[1] *= 1000.0;
                row[2] *= 1000.0;
            }
        }
        const auto bandwidthRows = ReadNumericRows(bandwidthTrace);
        const std::size_t count = std::min(bandwidthRows.size(), latencyRows.size());
        NS_ABORT_MSG_IF(count == 0, "The generated trace pair is empty");
        g_trace.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            g_trace.push_back({bandwidthRows[index][1],
                               bandwidthRows[index][2],
                               latencyRows[index][1],
                               latencyRows[index][2]});
        }
    }
    else
    {
        // The two local SIGCOMM columns are RTT components: their sum is the
        // end-to-end base RTT (about 22 ms in the paper's replayed BP trace).
        // Each ns-3 channel delay is one-way, so halve each component before
        // ns-3 traverses the path in both directions.
        LoadTrace(bandwidthTrace, latencyTrace, 0.5);
    }

    if (targetMinRttMs <= 0.0)
    {
        if (traceSet == "generated")
        {
            targetMinRttMs = pathMode == "BP" ? 50.0 : 100.0;
        }
        else
        {
            targetMinRttMs = 2.0 * (MinimumDynamicOneWayDelay() + 0.1);
        }
    }
    g_fixedDelayMs = std::max(0.1, targetMinRttMs / 2.0 - MinimumDynamicOneWayDelay());
    g_backgroundErrorRate = errorRate;

    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::" + tcpTypeId));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1 << 23));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 23));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(g_segmentSize));
    Config::SetDefault("ns3::TcpSocketState::EnablePacing", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketState::MaxPacingRate", DataRateValue(DataRate("4Gbps")));
    if (tcpTypeId == "TcpLeoCC")
    {
        Config::SetDefault("ns3::TcpLeoCC::EnableReconfiguration",
                           BooleanValue(enableReconfiguration));
        Config::SetDefault("ns3::TcpLeoCC::ReconfigurationInterval",
                           TimeValue(Seconds(reconfigurationInterval)));
        Config::SetDefault("ns3::TcpLeoCC::ReconfigurationOffset",
                           TimeValue(Seconds(reconfigurationOffset)));
    }

    NodeContainer nodes;
    nodes.Create(4);

    PointToPointHelper dynamicLink;
    dynamicLink.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    dynamicLink.SetChannelAttribute("Delay", StringValue("2ms"));
    dynamicLink.SetQueue("ns3::DropTailQueue<Packet>",
                         "MaxSize",
                         QueueSizeValue(QueueSize(std::to_string(deviceQueuePackets) + "p")));
    g_dishSatellite = dynamicLink.Install(nodes.Get(0), nodes.Get(1));
    g_satelliteGs = dynamicLink.Install(nodes.Get(1), nodes.Get(2));
    g_firstDeviceQueue = DynamicCast<PointToPointNetDevice>(g_dishSatellite.Get(0))->GetQueue();
    g_secondDeviceQueue = DynamicCast<PointToPointNetDevice>(g_satelliteGs.Get(0))->GetQueue();

    PointToPointHelper fixedLink;
    fixedLink.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    fixedLink.SetChannelAttribute("Delay", TimeValue(MilliSeconds(g_fixedDelayMs)));
    fixedLink.SetQueue("ns3::DropTailQueue<Packet>",
                       "MaxSize",
                       QueueSizeValue(QueueSize(std::to_string(deviceQueuePackets) + "p")));
    const auto gsPop = fixedLink.Install(nodes.Get(2), nodes.Get(3));

    g_handoverForwardErrorModel = CreateObject<RateErrorModel>();
    g_handoverForwardErrorModel->SetAttribute("ErrorRate", DoubleValue(g_backgroundErrorRate));
    g_handoverForwardErrorModel->SetAttribute("ErrorUnit",
                                               EnumValue(RateErrorModel::ERROR_UNIT_PACKET));
    g_handoverReverseErrorModel = CreateObject<RateErrorModel>();
    g_handoverReverseErrorModel->SetAttribute("ErrorRate", DoubleValue(0.0));
    g_handoverReverseErrorModel->SetAttribute("ErrorUnit",
                                               EnumValue(RateErrorModel::ERROR_UNIT_PACKET));
    // The terminal-satellite access link is unavailable in both directions.
    // The queue gate retains terminal-side packets while receive error models
    // invalidate every frame that overlaps the physical outage.
    g_dishSatellite.Get(1)->SetAttribute("ReceiveErrorModel",
                                      PointerValue(g_handoverForwardErrorModel));
    g_dishSatellite.Get(0)->SetAttribute("ReceiveErrorModel",
                                      PointerValue(g_handoverReverseErrorModel));

    InternetStackHelper internet;
    internet.Install(nodes);

    TrafficControlHelper trafficControl;
    trafficControl.SetRootQueueDisc("ns3::HandoverFifoQueueDisc",
                                    "MaxSize",
                                    QueueSizeValue(QueueSize(std::to_string(queuePackets) + "p")));
    g_firstQueue = trafficControl.Install(g_dishSatellite.Get(0)).Get(0);
    g_secondQueue = trafficControl.Install(g_satelliteGs.Get(0)).Get(0);
    g_handoverQueue = DynamicCast<HandoverFifoQueueDisc>(g_firstQueue);
    NS_ABORT_MSG_IF(!g_handoverQueue, "Missing handover-aware terminal queue disc");

    Ipv4AddressHelper addresses;
    addresses.SetBase("10.0.0.0", "255.255.255.0");
    addresses.Assign(g_dishSatellite);
    addresses.NewNetwork();
    addresses.Assign(g_satelliteGs);
    addresses.NewNetwork();
    const auto popInterfaces = addresses.Assign(gsPop);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    constexpr uint16_t port = 50001;
    PacketSinkHelper sinkHelper("ns3::TcpSocketFactory",
                                InetSocketAddress(Ipv4Address::GetAny(), port));
    auto sinkApplications = sinkHelper.Install(nodes.Get(3));
    sinkApplications.Start(Seconds(0));
    sinkApplications.Stop(Seconds(stopTime));
    g_sink = DynamicCast<PacketSink>(sinkApplications.Get(0));

    BulkSendHelper source("ns3::TcpSocketFactory",
                          InetSocketAddress(popInterfaces.GetAddress(1), port));
    source.SetAttribute("MaxBytes", UintegerValue(0));
    auto sourceApplications = source.Install(nodes.Get(0));
    sourceApplications.Start(Seconds(0.1));
    sourceApplications.Stop(Seconds(stopTime));

    OpenOutputs(outputDirectory);
    FlowMonitorHelper flowHelper;
    g_monitor = flowHelper.InstallAll();

    Simulator::ScheduleNow(&UpdateLinks);
    Simulator::Schedule(Seconds(0.101), &ConnectTcpTraces);
    Simulator::Schedule(Seconds(g_throughputPeriod), &TraceThroughput);
    Simulator::Schedule(Seconds(g_throughputPeriod), &TraceTxThroughput);
    Simulator::Schedule(Seconds(g_diagnosticPeriod), &TraceDiagnostics);
    if (enableHandover)
    {
        Simulator::Schedule(Seconds(handoverTime), &BeginHandover);
    }
    if (reconfigurationNotificationTime >= 0.0)
    {
        Simulator::Schedule(Seconds(reconfigurationNotificationTime),
                            &NotifyLeoCcReconfiguration);
    }
    Simulator::Stop(Seconds(stopTime) + TimeStep(1));
    Simulator::Run();
    g_monitor->CheckForLostPackets();
    const uint64_t finalRxBytes = g_sink ? g_sink->GetTotalRx() : 0;
    uint64_t finalTxBytes = 0;
    uint64_t finalTxPackets = 0;
    uint64_t finalLostPackets = 0;
    for (const auto& [flowId, stats] : g_monitor->GetFlowStats())
    {
        (void)flowId;
        if (stats.txBytes > finalTxBytes)
        {
            finalTxBytes = stats.txBytes;
            finalTxPackets = stats.txPackets;
            finalLostPackets = stats.lostPackets;
        }
    }
    g_loss << stopTime << " "
           << (finalTxPackets > 0 ? static_cast<double>(finalLostPackets) / finalTxPackets : 0.0)
           << std::endl;
    Simulator::Destroy();

    std::ofstream metadata(outputDirectory + "/metadata.txt");
    metadata << "tcp=" << tcpTypeId << "\n"
             << "trace_set=" << traceSet << "\n"
             << "path_mode=" << pathMode << "\n"
             << "stop_time_s=" << stopTime << "\n"
             << "throughput_period_s=" << g_throughputPeriod << "\n"
             << "diagnostic_period_s=" << g_diagnosticPeriod << "\n"
             << "bandwidth_jitter_distribution=normal_multiplicative\n"
             << "bandwidth_jitter_mean=0\n"
             << "bandwidth_jitter_std=" << g_bandwidthJitterStd << "\n"
             << "bandwidth_jitter_seed=" << jitterSeed << "\n"
             << "link_update_period_s=" << g_linkUpdatePeriod << "\n"
             << "latency_trace_semantics="
             << (traceSet == "sigcomm" ? "columns_sum_to_end_to_end_base_rtt"
                                        : "one_way_per_dynamic_link")
             << "\n"
             << "latency_channel_scale=" << (traceSet == "sigcomm" ? 0.5 : 1.0) << "\n"
             << "target_min_rtt_ms=" << targetMinRttMs << "\n"
             << "fixed_link_delay_ms=" << g_fixedDelayMs << "\n"
             << "error_rate=" << errorRate << "\n"
             << "queue_packets=" << queuePackets << "\n"
             << "device_queue_packets=" << deviceQueuePackets << "\n"
             << "reconfiguration=" << enableReconfiguration << "\n"
             << "reconfiguration_interval_s=" << reconfigurationInterval << "\n"
             << "reconfiguration_offset_s=" << reconfigurationOffset << "\n"
             << "handover=" << enableHandover << "\n"
             << "handover_time_s=" << handoverTime << "\n"
             << "handover_duration_ms=" << g_handoverDurationMs << "\n"
             << "handover_outage_model=bidirectional_terminal_satellite_blackout_with_tx_gate\n"
             << "handover_old_satellite_queue_policy=flush_at_start_and_end\n"
             << "reconfiguration_notification_time_s=" << reconfigurationNotificationTime << "\n"
             << "throughput_dat=receiver_goodput\n"
             << "tx_throughput_dat=sender_flowmonitor_throughput\n"
             << "final_rx_bytes=" << finalRxBytes << "\n"
             << "final_tx_bytes=" << finalTxBytes << "\n"
             << "final_tx_packets=" << finalTxPackets << "\n"
             << "final_lost_packets=" << finalLostPackets << "\n";
    return 0;
}
