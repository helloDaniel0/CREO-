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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
using namespace ns3::SystemPath;

namespace
{
std::string g_dir;
std::ofstream g_throughput;
std::ofstream g_queueSize;
std::ofstream g_realBw;
uint64_t g_prevRx = 0;
Time g_prevTime = Seconds(0);
const uint32_t g_mtuBytes = 1000;

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
    uint32_t line = static_cast<uint32_t>(std::floor((now + 100.0) * 10.0)) % 5731;
    double bwMbps = std::max(1.0, TraceOrSynthetic(bwTrace, line, 1, now, 32.0, 12.0));
    double delayMs = std::max(1.0, TraceOrSynthetic(latencyTrace, line, 1, now, 2.2, 0.4));
    Ptr<PointToPointChannel> channel = bottleneck.Get(0)->GetChannel()->GetObject<PointToPointChannel>();
    if (channel)
    {
        channel->SetAttribute("Delay", StringValue(std::to_string(delayMs) + "ms"));
    }
    for (uint32_t i = 0; i < bottleneck.GetN(); ++i)
    {
        Ptr<PointToPointNetDevice> dev = bottleneck.Get(i)->GetObject<PointToPointNetDevice>();
        if (dev)
        {
            dev->SetAttribute("DataRate", StringValue(std::to_string(bwMbps) + "Mbps"));
        }
    }
    for (auto socket : GetTcpSockets())
    {
        socket->SetCapacityLatency(bwMbps, delayMs + processDelayMs);
    }
    g_realBw << now << " " << bwMbps << " " << 2.0 * (delayMs + processDelayMs) << std::endl;
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
} // namespace

int
main(int argc, char* argv[])
{
    std::string tcpTypeId = "TcpRlTimeBased";
    double duration = 30.0;
    uint32_t simSeed = 1;
    uint32_t flows = 3;
    bool enableHandover = false;
    double handoverInterval = 15.0;
    double handoverDuration = 0.05;
    double errorRate = 0.0;
    std::string bwTrace = "dataset/SIGCOMMbw.txt";
    std::string latencyTrace = "dataset/SIGCOMMlatency.txt";
    double processDelayMs = 3.0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("transport_prot", "TCP type", tcpTypeId);
    cmd.AddValue("duration", "Simulation duration in seconds", duration);
    cmd.AddValue("simSeed", "ns-3 RNG run", simSeed);
    cmd.AddValue("flows", "Number of competing TCP flows", flows);
    cmd.AddValue("enableHandover", "Enable periodic packet-loss handover events", enableHandover);
    cmd.AddValue("handoverInterval", "Seconds between handover starts", handoverInterval);
    cmd.AddValue("handoverDuration", "Handover interruption duration", handoverDuration);
    cmd.AddValue("bwTrace", "Capacity trace path relative to ns-3 root", bwTrace);
    cmd.AddValue("latencyTrace", "Latency trace path relative to ns-3 root", latencyTrace);
    cmd.Parse(argc, argv);
    flows = std::max(1u, flows);

    RngSeedManager::SetRun(simSeed);
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::" + tcpTypeId));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1 << 23));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 23));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(g_mtuBytes));
    Config::SetDefault("ns3::TcpSocketState::EnablePacing", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocketState::MaxPacingRate", DataRateValue(DataRate("4Gbps")));

    Ptr<OpenGymInterface> openGymInterface;
    if (tcpTypeId == "TcpRlTimeBased" || tcpTypeId == "TcpRlEventBased")
    {
        openGymInterface = OpenGymInterface::Get();
    }

    NodeContainer dishes;
    NodeContainer leo;
    NodeContainer gs;
    NodeContainer pop;
    dishes.Create(flows);
    leo.Create(1);
    gs.Create(1);
    pop.Create(1);

    PointToPointHelper access;
    access.SetDeviceAttribute("DataRate", StringValue("400Mbps"));
    access.SetChannelAttribute("Delay", StringValue("2ms"));
    PointToPointHelper bottleneckHelper;
    bottleneckHelper.SetDeviceAttribute("DataRate", StringValue("32Mbps"));
    bottleneckHelper.SetChannelAttribute("Delay", StringValue("2ms"));
    bottleneckHelper.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize("2000p")));
    PointToPointHelper terrestrial;
    terrestrial.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    terrestrial.SetChannelAttribute("Delay", StringValue("1ms"));

    std::vector<NetDeviceContainer> dishLeo(flows);
    for (uint32_t i = 0; i < flows; ++i)
    {
        dishLeo[i] = access.Install(dishes.Get(i), leo.Get(0));
    }
    NetDeviceContainer leoGs = bottleneckHelper.Install(leo.Get(0), gs.Get(0));
    NetDeviceContainer gsPop = terrestrial.Install(gs.Get(0), pop.Get(0));

    InternetStackHelper internet;
    internet.Install(dishes);
    internet.Install(leo);
    internet.Install(gs);
    internet.Install(pop);

    Ipv4AddressHelper ipv4;
    for (uint32_t i = 0; i < flows; ++i)
    {
        std::ostringstream subnet;
        subnet << "10." << (i + 1) << ".0.0";
        ipv4.SetBase(subnet.str().c_str(), "255.255.255.0");
        ipv4.Assign(dishLeo[i]);
    }
    ipv4.SetBase("172.16.0.0", "255.255.255.0");
    ipv4.Assign(leoGs);
    ipv4.SetBase("192.168.0.0", "255.255.255.0");
    Ipv4InterfaceContainer popIf = ipv4.Assign(gsPop);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    for (uint32_t i = 0; i < flows; ++i)
    {
        uint16_t port = 50001 + i;
        PacketSinkHelper sink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sink.Install(pop.Get(0));
        sinkApp.Start(Seconds(0.0));
        sinkApp.Stop(Seconds(duration));

        BulkSendHelper source("ns3::TcpSocketFactory", InetSocketAddress(popIf.GetAddress(1), port));
        source.SetAttribute("MaxBytes", UintegerValue(0));
        ApplicationContainer sourceApp = source.Install(dishes.Get(i));
        sourceApp.Start(Seconds(0.1 + 0.02 * i));
        sourceApp.Stop(Seconds(duration));
    }

    time_t rawtime;
    char buffer[80];
    time(&rawtime);
    strftime(buffer, sizeof(buffer), "%d-%m-%Y-%H-%M-%S", localtime(&rawtime));
    g_dir = "results/creo-multi-" + std::to_string(flows) + "-" + tcpTypeId + "-" + std::string(buffer);
    MakeDirectories(g_dir);
    g_throughput.open(g_dir + "/throughput.dat");
    g_queueSize.open(g_dir + "/queueSize.dat");
    g_realBw.open(g_dir + "/realbw.dat");

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();
    Ptr<PointToPointNetDevice> bottleneckDev = DynamicCast<PointToPointNetDevice>(leoGs.Get(0));
    Simulator::Schedule(Seconds(0.101), &TraceThroughput, monitor);
    Simulator::Schedule(Seconds(0.0), &TraceQueue, bottleneckDev->GetQueue());
    Simulator::Schedule(Seconds(0.101), &UpdateLeoLink, leoGs, bwTrace, latencyTrace, processDelayMs);
    if (enableHandover)
    {
        Simulator::Schedule(Seconds(handoverInterval), &ScheduleHandover, leoGs, handoverInterval, handoverDuration, errorRate);
    }

    Simulator::Stop(Seconds(duration) + TimeStep(1));
    Simulator::Run();
    if (openGymInterface)
    {
        openGymInterface->NotifySimulationEnd();
    }
    Simulator::Destroy();
    g_throughput.close();
    g_queueSize.close();
    g_realBw.close();
    return 0;
}
