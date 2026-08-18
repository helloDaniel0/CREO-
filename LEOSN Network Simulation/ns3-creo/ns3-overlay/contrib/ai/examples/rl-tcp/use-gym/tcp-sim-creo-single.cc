#include "ns3/ai-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <x86intrin.h>

using namespace ns3;
using namespace ns3::SystemPath;

namespace
{
std::string g_dir;
std::ofstream g_throughput;
std::ofstream g_queueSize;
std::ofstream g_realBw;
std::ofstream g_prop;
uint64_t g_prevRx = 0;
Time g_prevTime = Seconds(0);
const uint32_t g_mtuBytes = 1000;
std::vector<double> g_bwTraceValues;
std::vector<double> g_latencyTraceValues;
std::string g_measuredTcpType;
uint64_t g_coreCycles = 0;
uint64_t g_coreCalls = 0;
uint64_t g_cycleReadOverhead = 0;
Time g_measureStart = Seconds(0);
Time g_measureEnd = Time::Max();

uint64_t
CycleStart()
{
    _mm_lfence();
    return __rdtsc();
}

uint64_t
CycleEnd()
{
    unsigned int auxiliary;
    uint64_t cycles = __rdtscp(&auxiliary);
    _mm_lfence();
    return cycles;
}

void
CalibrateCycleRead()
{
    g_cycleReadOverhead = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < 10000; ++i)
    {
        uint64_t start = CycleStart();
        uint64_t end = CycleEnd();
        g_cycleReadOverhead = std::min(g_cycleReadOverhead, end - start);
    }
}

template <typename Callback>
auto
MeasureCore(Callback&& callback)
{
    const Time now = Simulator::Now();
    if (now < g_measureStart || now >= g_measureEnd)
    {
        if constexpr (std::is_void_v<std::invoke_result_t<Callback>>)
        {
            callback();
            return;
        }
        else
        {
            return callback();
        }
    }

    uint64_t start = CycleStart();
    if constexpr (std::is_void_v<std::invoke_result_t<Callback>>)
    {
        callback();
        uint64_t elapsed = CycleEnd() - start;
        g_coreCycles += elapsed > g_cycleReadOverhead ? elapsed - g_cycleReadOverhead : 0;
        ++g_coreCalls;
    }
    else
    {
        auto result = callback();
        uint64_t elapsed = CycleEnd() - start;
        g_coreCycles += elapsed > g_cycleReadOverhead ? elapsed - g_cycleReadOverhead : 0;
        ++g_coreCalls;
        return result;
    }
}

class TcpCcaCycleProbe : public TcpCongestionOps
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TcpCcaCycleProbe")
                                .SetParent<TcpCongestionOps>()
                                .SetGroupName("Internet")
                                .AddConstructor<TcpCcaCycleProbe>();
        return tid;
    }

    TcpCcaCycleProbe()
    {
        NS_ABORT_MSG_IF(g_measuredTcpType.empty(), "measured TCP type was not configured");
        ObjectFactory factory;
        factory.SetTypeId("ns3::" + g_measuredTcpType);
        m_inner = factory.Create<TcpCongestionOps>();
    }

    TcpCcaCycleProbe(const TcpCcaCycleProbe& other)
        : TcpCongestionOps(other),
          m_inner(other.m_inner->Fork())
    {
    }

    std::string GetName() const override
    {
        return m_inner->GetName();
    }

    void Init(Ptr<TcpSocketState> tcb) override
    {
        MeasureCore([&]() { m_inner->Init(tcb); });
    }

    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override
    {
        return MeasureCore([&]() { return m_inner->GetSsThresh(tcb, bytesInFlight); });
    }

    void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override
    {
        MeasureCore([&]() { m_inner->IncreaseWindow(tcb, segmentsAcked); });
    }

    void PktsAcked(Ptr<TcpSocketState> tcb,
                   uint32_t segmentsAcked,
                   const Time& rtt) override
    {
        MeasureCore([&]() { m_inner->PktsAcked(tcb, segmentsAcked, rtt); });
    }

    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override
    {
        MeasureCore([&]() { m_inner->CongestionStateSet(tcb, newState); });
    }

    void CwndEvent(Ptr<TcpSocketState> tcb,
                   const TcpSocketState::TcpCAEvent_t event) override
    {
        MeasureCore([&]() { m_inner->CwndEvent(tcb, event); });
    }

    bool HasCongControl() const override
    {
        // This query selects the TCP call path; it is not a CCA computation.
        return m_inner->HasCongControl();
    }

    void CongControl(Ptr<TcpSocketState> tcb,
                     const TcpRateOps::TcpRateConnection& rc,
                     const TcpRateOps::TcpRateSample& rs) override
    {
        MeasureCore([&]() { m_inner->CongControl(tcb, rc, rs); });
    }

    Ptr<TcpCongestionOps> Fork() override
    {
        return CopyObject<TcpCcaCycleProbe>(this);
    }

  private:
    Ptr<TcpCongestionOps> m_inner;
};

NS_OBJECT_ENSURE_REGISTERED(TcpCcaCycleProbe);

std::vector<std::string>
ReadTraceLine(const std::string& filename, uint32_t lineNumber)
{
    std::ifstream file(filename);
    std::string line;
    uint32_t current = 0;
    while (std::getline(file, line))
    {
        if (current == lineNumber)
        {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token)
            {
                tokens.push_back(token);
            }
            return tokens;
        }
        current++;
    }
    return {};
}

std::vector<double>
ReadTraceColumn(const std::string& filename, uint32_t column)
{
    std::ifstream file(filename);
    std::vector<double> values;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string token;
        uint32_t current = 0;
        while (iss >> token)
        {
            if (current++ == column)
            {
                try
                {
                    double value = std::stod(token);
                    if (std::isfinite(value) && value > 0.0)
                    {
                        values.push_back(value);
                    }
                }
                catch (const std::exception&)
                {
                }
                break;
            }
        }
    }
    return values;
}

double
Round2(double value)
{
    return std::round(value * 100.0) / 100.0;
}

double
TraceOrSynthetic(const std::string& filename, uint32_t line, uint32_t col, double now, double base, double amp)
{
    auto tokens = ReadTraceLine(filename, line);
    if (tokens.size() > col)
    {
        return std::stod(tokens[col]);
    }
    return base + amp * std::sin(now / 8.0) + amp * 0.25 * std::sin(now / 1.7);
}

std::vector<Ptr<TcpSocketBase>>
GetTcpSockets()
{
    std::vector<Ptr<TcpSocketBase>> sockets;
    for (NodeList::Iterator nodeIt = NodeList::Begin(); nodeIt != NodeList::End(); ++nodeIt)
    {
        Ptr<TcpL4Protocol> tcp = (*nodeIt)->GetObject<TcpL4Protocol>();
        if (!tcp)
        {
            continue;
        }
        ObjectVectorValue socketVec;
        tcp->GetAttribute("SocketList", socketVec);
        for (uint32_t i = 0; i < socketVec.GetN(); ++i)
        {
            Ptr<TcpSocketBase> socket = DynamicCast<TcpSocketBase>(socketVec.Get(i));
            if (socket)
            {
                sockets.push_back(socket);
            }
        }
    }
    return sockets;
}

void
TraceThroughput(Ptr<FlowMonitor> monitor)
{
    uint64_t rxBytes = 0;
    for (const auto& item : monitor->GetFlowStats())
    {
        rxBytes += item.second.rxBytes;
    }
    Time now = Simulator::Now();
    double mbps = 0.0;
    if (now > g_prevTime)
    {
        mbps = 8.0 * (rxBytes - g_prevRx) / (now - g_prevTime).GetMicroSeconds();
    }
    g_throughput << now.GetSeconds() << " " << mbps << std::endl;
    g_prevRx = rxBytes;
    g_prevTime = now;
    Simulator::Schedule(Seconds(0.1), &TraceThroughput, monitor);
}

void
TraceQueue(Ptr<Queue<Packet>> queue)
{
    g_queueSize << Simulator::Now().GetSeconds() << " " << queue->GetNPackets() << std::endl;
    Simulator::Schedule(Seconds(0.02), &TraceQueue, queue);
}

void
UpdateLeoLink(NetDeviceContainer bottleneck,
              const std::string& bwTrace,
              const std::string& latencyTrace,
              double processDelayMs)
{
    double now = Simulator::Now().GetSeconds();
    uint64_t sample = static_cast<uint64_t>(std::floor((now + 100.0) * 10.0));
    double rawBw = g_bwTraceValues.empty()
                       ? TraceOrSynthetic(bwTrace, sample, 1, now, 32.0, 12.0)
                       : g_bwTraceValues[sample % g_bwTraceValues.size()];
    double rawDelay = g_latencyTraceValues.empty()
                          ? TraceOrSynthetic(latencyTrace, sample, 1, now, 2.2, 0.4)
                          : g_latencyTraceValues[sample % g_latencyTraceValues.size()];
    // Topology-derived latency traces use seconds; SIGCOMM traces use ms.
    if (rawDelay > 0.0 && rawDelay < 0.1)
    {
        rawDelay *= 1000.0;
    }
    double bwMbps = std::max(1.0, Round2(rawBw));
    double delayMs = std::max(1.0, Round2(rawDelay));
    double rtpropMs = 2.0 * (delayMs + processDelayMs);

    for (uint32_t i = 0; i < bottleneck.GetN(); ++i)
    {
        Ptr<PointToPointNetDevice> dev = bottleneck.Get(i)->GetObject<PointToPointNetDevice>();
        if (dev)
        {
            dev->SetAttribute("DataRate", StringValue(std::to_string(bwMbps) + "Mbps"));
        }
    }
    Ptr<PointToPointChannel> channel = bottleneck.Get(0)->GetChannel()->GetObject<PointToPointChannel>();
    if (channel)
    {
        channel->SetAttribute("Delay", StringValue(std::to_string(delayMs) + "ms"));
    }
    for (auto socket : GetTcpSockets())
    {
        socket->SetCapacityLatency(bwMbps, delayMs + processDelayMs);
    }
    g_realBw << now << " " << bwMbps << " " << rtpropMs << std::endl;
    g_prop << now << " " << rtpropMs << std::endl;

    Simulator::Schedule(Seconds(0.1), &UpdateLeoLink, bottleneck, bwTrace, latencyTrace, processDelayMs);
}

void
SetHandover(NetDeviceContainer bottleneck, bool active, double errorRate)
{
    Ptr<RateErrorModel> em = CreateObjectWithAttributes<RateErrorModel>(
        "ErrorRate",
        DoubleValue(active ? 1.0 : errorRate),
        "ErrorUnit",
        EnumValue(RateErrorModel::ERROR_UNIT_PACKET));
    for (uint32_t i = 0; i < bottleneck.GetN(); ++i)
    {
        bottleneck.Get(i)->SetAttribute("ReceiveErrorModel", PointerValue(em));
    }
}

void
ScheduleHandover(NetDeviceContainer bottleneck,
                 double interval,
                 double duration,
                 double errorRate)
{
    SetHandover(bottleneck, true, errorRate);
    Simulator::Schedule(Seconds(duration), &SetHandover, bottleneck, false, errorRate);
    Simulator::Schedule(Seconds(interval), &ScheduleHandover, bottleneck, interval, duration, errorRate);
}

void
WriteCwnd(Ptr<OutputStreamWrapper> stream, uint32_t, uint32_t newValue)
{
    *stream->GetStream() << Simulator::Now().GetSeconds() << " " << newValue / g_mtuBytes
                         << std::endl;
}

void
TraceCwnd(uint32_t nodeId, uint32_t socketId)
{
    AsciiTraceHelper ascii;
    Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream(g_dir + "/cwnd.dat");
    Config::ConnectWithoutContext(
        "/NodeList/" + std::to_string(nodeId) + "/$ns3::TcpL4Protocol/SocketList/" +
            std::to_string(socketId) + "/CongestionWindow",
        MakeBoundCallback(&WriteCwnd, stream));
}
} // namespace

int
main(int argc, char* argv[])
{
    std::string tcpTypeId = "TcpRlTimeBased";
    double duration = 30.0;
    uint32_t simSeed = 1;
    uint32_t flows = 1;
    bool enableHandover = false;
    double handoverInterval = 15.0;
    double handoverDuration = 0.05;
    double errorRate = 0.0;
    std::string bwTrace = "dataset/SIGCOMMbw.txt";
    std::string latencyTrace = "dataset/SIGCOMMlatency.txt";
    double processDelayMs = 3.0;
    bool burstPacing = true;
    uint32_t miniWindow = 15;
    bool adaptiveSp = true;
    double spCapMs = 30.0;
    uint32_t initialCwnd = 10;
    uint32_t queuePackets = 2000;
    bool measureCoreCycles = false;
    double measureStart = 0.0;
    double measureEnd = -1.0;
    std::string outputDir;

    CommandLine cmd(__FILE__);
    cmd.AddValue("transport_prot", "TCP type, e.g., TcpRlTimeBased, TcpBbr, TcpCubic", tcpTypeId);
    cmd.AddValue("duration", "Simulation duration in seconds", duration);
    cmd.AddValue("simSeed", "ns-3 RNG run", simSeed);
    cmd.AddValue("flows", "Accepted for script compatibility; single target always uses one flow", flows);
    cmd.AddValue("enableHandover", "Enable periodic packet-loss handover events", enableHandover);
    cmd.AddValue("handoverInterval", "Seconds between handover starts", handoverInterval);
    cmd.AddValue("handoverDuration", "Handover interruption duration", handoverDuration);
    cmd.AddValue("bwTrace", "Capacity trace path relative to ns-3 root", bwTrace);
    cmd.AddValue("latencyTrace", "Latency trace path relative to ns-3 root", latencyTrace);
    cmd.AddValue("burstPacing",
                 "Use burst mini-window ACK dispersion instead of SP throughput sampling",
                 burstPacing);
    cmd.AddValue("miniWindow", "Segments in one burst-pacing capacity sample", miniWindow);
    cmd.AddValue("adaptiveSp", "Use the paper's RTT-adaptive statistical period", adaptiveSp);
    cmd.AddValue("spCapMs", "DeltaT0 upper bound of the statistical period", spCapMs);
    cmd.AddValue("initialCwnd", "Initial congestion window in segments", initialCwnd);
    cmd.AddValue("queuePackets", "Bottleneck device queue in packets", queuePackets);
    cmd.AddValue("measureCoreCycles",
                 "Wrap the selected native CCA and count only its callbacks",
                 measureCoreCycles);
    cmd.AddValue("measureStart",
                 "Start time for native CCA cycle accounting, in seconds",
                 measureStart);
    cmd.AddValue("measureEnd",
                 "Exclusive end time for native CCA cycle accounting, in seconds",
                 measureEnd);
    cmd.AddValue("outputDir", "Result directory; empty uses a timestamped path", outputDir);
    cmd.Parse(argc, argv);

    g_bwTraceValues = ReadTraceColumn(bwTrace, 1);
    g_latencyTraceValues = ReadTraceColumn(latencyTrace, 1);

    RngSeedManager::SetRun(simSeed);
    if (measureCoreCycles)
    {
        NS_ABORT_MSG_IF(tcpTypeId == "TcpRlTimeBased" || tcpTypeId == "TcpRlEventBased",
                        "native CCA cycle probe cannot wrap the Gym CCA");
        g_measuredTcpType = tcpTypeId;
        g_measureStart = Seconds(std::max(0.0, measureStart));
        g_measureEnd = measureEnd > 0.0 ? Seconds(std::max(measureStart, measureEnd))
                                        : Time::Max();
        CalibrateCycleRead();
    }
    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       StringValue(measureCoreCycles ? "ns3::TcpCcaCycleProbe"
                                                     : "ns3::" + tcpTypeId));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1 << 23));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 23));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(initialCwnd));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(g_mtuBytes));
    Config::SetDefault("ns3::TcpSocketState::EnablePacing", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketState::MaxPacingRate", DataRateValue(DataRate("4Gbps")));
    Config::SetDefault("ns3::TcpTimeStepEnv::EnableBurstPacingSample",
                       BooleanValue(burstPacing));
    Config::SetDefault("ns3::TcpTimeStepEnv::MiniWindowSegments", UintegerValue(miniWindow));
    Config::SetDefault("ns3::TcpTimeStepEnv::AdaptiveStep", BooleanValue(adaptiveSp));
    Config::SetDefault("ns3::TcpTimeStepEnv::StepTimeCap",
                       TimeValue(MilliSeconds(spCapMs)));
    Config::SetDefault("ns3::TcpTimeStepEnv::StepTime",
                       TimeValue(MilliSeconds(spCapMs)));

    Ptr<OpenGymInterface> openGymInterface;
    if (tcpTypeId == "TcpRlTimeBased" || tcpTypeId == "TcpRlEventBased")
    {
        openGymInterface = OpenGymInterface::Get();
    }

    NodeContainer dish;
    NodeContainer leo;
    NodeContainer gs;
    NodeContainer pop;
    dish.Create(1);
    leo.Create(1);
    gs.Create(1);
    pop.Create(1);

    PointToPointHelper access;
    access.SetDeviceAttribute("DataRate", StringValue("400Mbps"));
    access.SetChannelAttribute("Delay", StringValue("2ms"));
    PointToPointHelper bottleneckHelper;
    bottleneckHelper.SetDeviceAttribute("DataRate", StringValue("32Mbps"));
    bottleneckHelper.SetChannelAttribute("Delay", StringValue("2ms"));
    bottleneckHelper.SetQueue(
        "ns3::DropTailQueue<Packet>",
        "MaxSize",
        QueueSizeValue(QueueSize(std::to_string(queuePackets) + "p")));
    PointToPointHelper terrestrial;
    terrestrial.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    terrestrial.SetChannelAttribute("Delay", StringValue("1ms"));

    NetDeviceContainer dishLeo = access.Install(dish.Get(0), leo.Get(0));
    NetDeviceContainer leoGs = bottleneckHelper.Install(leo.Get(0), gs.Get(0));
    NetDeviceContainer gsPop = terrestrial.Install(gs.Get(0), pop.Get(0));

    InternetStackHelper internet;
    internet.Install(dish);
    internet.Install(leo);
    internet.Install(gs);
    internet.Install(pop);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.0.0", "255.255.255.0");
    ipv4.Assign(dishLeo);
    ipv4.SetBase("10.2.0.0", "255.255.255.0");
    ipv4.Assign(leoGs);
    ipv4.SetBase("10.3.0.0", "255.255.255.0");
    Ipv4InterfaceContainer popIf = ipv4.Assign(gsPop);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    uint16_t port = 50001;
    PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sink.Install(pop.Get(0));
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(duration));

    BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(popIf.GetAddress(1), port));
    source.SetAttribute("MaxBytes", UintegerValue(0));
    ApplicationContainer sourceApp = source.Install(dish.Get(0));
    sourceApp.Start(Seconds(0.1));
    sourceApp.Stop(Seconds(duration));

    time_t rawtime;
    char buffer[80];
    time(&rawtime);
    strftime(buffer, sizeof(buffer), "%d-%m-%Y-%H-%M-%S", localtime(&rawtime));
    g_dir = outputDir.empty()
                ? "results/creo-single-" + tcpTypeId + "-" + std::string(buffer)
                : outputDir;
    MakeDirectories(g_dir);
    g_throughput.open(g_dir + "/throughput.dat");
    g_queueSize.open(g_dir + "/queueSize.dat");
    g_realBw.open(g_dir + "/realbw.dat");
    g_prop.open(g_dir + "/prop.dat");

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();
    Ptr<PointToPointNetDevice> bottleneckDev = DynamicCast<PointToPointNetDevice>(leoGs.Get(0));
    Simulator::Schedule(Seconds(0.101), &TraceThroughput, monitor);
    Simulator::Schedule(Seconds(0.101), &TraceCwnd, 0, 0);
    Simulator::Schedule(Seconds(0.0), &TraceQueue, bottleneckDev->GetQueue());
    Simulator::Schedule(Seconds(0.101), &UpdateLeoLink, leoGs, bwTrace, latencyTrace, processDelayMs);
    if (enableHandover)
    {
        Simulator::Schedule(Seconds(handoverInterval), &ScheduleHandover, leoGs, handoverInterval, handoverDuration, errorRate);
    }

    Simulator::Stop(Seconds(duration) + TimeStep(1));
    Simulator::Run();
    monitor->CheckForLostPackets();
    uint64_t finalRxBytes = 0;
    uint64_t finalTxPackets = 0;
    uint64_t finalLostPackets = 0;
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    for (const auto& [flowId, stats] : monitor->GetFlowStats())
    {
        Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow(flowId);
        if (tuple.destinationPort == port)
        {
            finalRxBytes += stats.rxBytes;
            finalTxPackets += stats.txPackets;
            finalLostPackets += stats.lostPackets;
        }
    }
    std::ofstream metadata(g_dir + "/metadata.txt");
    metadata << "tcp=" << tcpTypeId << "\n"
             << "duration_s=" << duration << "\n"
             << "adaptive_sp=" << adaptiveSp << "\n"
             << "sp_cap_ms=" << spCapMs << "\n"
             << "core_reference_cycles=" << g_coreCycles << "\n"
             << "core_callback_calls=" << g_coreCalls << "\n"
             << "cycle_read_overhead=" << g_cycleReadOverhead << "\n"
             << "measure_start_s=" << measureStart << "\n"
             << "measure_end_s=" << measureEnd << "\n"
             << "final_rx_bytes=" << finalRxBytes << "\n"
             << "final_tx_packets=" << finalTxPackets << "\n"
             << "final_lost_packets=" << finalLostPackets << "\n";
    metadata.close();
    if (openGymInterface)
    {
        openGymInterface->NotifySimulationEnd();
    }
    Simulator::Destroy();
    g_throughput.close();
    g_queueSize.close();
    g_realBw.close();
    g_prop.close();
    return 0;
}
