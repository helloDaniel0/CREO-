#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/error-model.h"
#include "ns3/quic-module.h"
#include "ns3/tcp-header.h"
#include "ns3/udp-header.h"

#include <filesystem>
#include <random>
#include <cmath>
#include <sys/stat.h>

using namespace ns3;
using namespace ns3::SystemPath;

std::string dir;
std::ofstream throughput;
std::ofstream wbw;
std::ofstream queueSize;

//星历模型
std::string starHis = "gw0-5";
std::string bw_filename = "./dataset/" + starHis + "/bw.txt";
std::string latency_filename = "./dataset/" + starHis + "/latency.txt";
uint32_t starStart = 10;    //星历读取起始位置

uint32_t mtu_bytes = 1460;  //包大小
double error_rate = 0.001;  //丢包率
bool isSeamless = false;    //是否使用无缝切换
uint32_t satSerUser = 1;    //卫星服务用户数量，卫星负载

bool isRain = false;        //降雨
float rainTimeStart = 50;
float rainTimeStop = 150;

uint64_t file_size = 0;     //文件大小
bool upLoad = true;         //上传还是下载
uint32_t bufferi = 24;      //收发队列
std::string satQueue = "500p";      //卫星，网关队列大小

uint64_t prev = 0;
Time prevTime = Seconds(0);


// Calculate throughput
static void
TraceThroughput(Ptr<FlowMonitor> monitor)
{
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    if (!stats.empty())
    {
        auto itr = stats.begin();
        Time curTime = Now();
        throughput << curTime.GetSeconds() << " "
                   << 8 * (itr->second.rxBytes - prev) / ((curTime - prevTime).ToDouble(Time::US))
                   << std::endl;
        prevTime = curTime;
        prev = itr->second.rxBytes;
    }
    Simulator::Schedule(Seconds(0.1), &TraceThroughput, monitor);
}

// Check the queue size
void
CheckQueueSize(Ptr<QueueDisc> qd)
{
    uint32_t qsize = qd->GetCurrentSize().GetValue();
    Simulator::Schedule(Seconds(0.2), &CheckQueueSize, qd);
    queueSize << Simulator::Now().GetSeconds() << " " << qsize << std::endl;
}

// Trace congestion window
static void
CwndTracer(Ptr<OutputStreamWrapper> stream, uint32_t oldval, uint32_t newval)
{
    *stream->GetStream() << Simulator::Now().GetSeconds() << " " << newval / 1448.0 << std::endl;
}

static void 
RttTracer(Ptr<OutputStreamWrapper> stream, Time oldRtt, Time newRtt)
{
  *stream->GetStream() << Simulator::Now().GetSeconds() << "\t" << newRtt.GetMilliSeconds() << std::endl;
}

void
TraceCwnd(uint32_t nodeId, uint32_t socketId)
{
    AsciiTraceHelper ascii;
    Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream(dir + "/cwnd.dat");
    Config::ConnectWithoutContext("/NodeList/" + std::to_string(nodeId) +
                                      "/$ns3::QuicL4Protocol/SocketList/" +
                                      std::to_string(socketId) + "/QuicSocketBase/CongestionWindow",
                                  MakeBoundCallback(&CwndTracer, stream));
}

void 
TraceRTT(uint32_t nodeId, uint32_t socketId)
{
  AsciiTraceHelper ascii;
  Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream(dir + "/rtt.dat");
  Config::ConnectWithoutContext("/NodeList/" + std::to_string(nodeId) +
                                    "/$ns3::QuicL4Protocol/SocketList/" +
                                    std::to_string(socketId) + "/QuicSocketBase/RTT",
                                MakeBoundCallback(&RttTracer, stream));
}

double roundToDecimal(double value, int decimalPlaces) {
    double factor = std::pow(10.0, decimalPlaces);
    return std::round(value * factor) / factor;
}

std::vector<std::string> ReadSpecificLineAndSplit(const std::string& filename, int lineNumber) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    std::string line;
    int currentLine = 0;
    while (std::getline(file, line)) {
        if (currentLine == lineNumber) {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
            return tokens;
        }
        currentLine++;
    }

    std::cerr << "Line number " << lineNumber << " not found in file: " << filename << std::endl;
    return {};
}

// void BindTcpSocket(NodeContainer tcpsender, Ptr<QuicL4Protocol> &tcp, Ptr<QuicSocketBase> &tcpsb){
//     tcp = tcpsender.Get(0)->GetObject<QuicL4Protocol>();
//     std::vector< Ptr<QuicUdpBinding> > sockets = tcp->GetSockets();
//     tcpsb = sockets[0]->m_quicSocket;
//     if(tcpsb == nullptr){
//         std::cout << "NULL" << std::endl;
//     }else{
//         std::cout <<tcpsb << " cwnd = " << tcpsb->GetCwnd() << std::endl;
//     }
// }

void PhysicalLayerHandover(NetDeviceContainer sat_gs, bool start){
    Ptr<NetDevice> relay = sat_gs.Get(0);

    if(start == true)
    {
    Ptr<RateErrorModel> em = CreateObjectWithAttributes<RateErrorModel>("ErrorRate", DoubleValue(1.0), 
                            "ErrorUnit", EnumValue(RateErrorModel::ERROR_UNIT_PACKET));
    relay->SetAttribute("ReceiveErrorModel", PointerValue(em));
    relay->GetObject<PointToPointNetDevice> ()->SetAttribute ("DataRate", StringValue("10kbps"));
    wbw << Simulator::Now().GetSeconds() << " "<< 0 << std::endl;
    }else{
    Ptr<RateErrorModel> em = CreateObjectWithAttributes<RateErrorModel>("ErrorRate", DoubleValue(error_rate), 
                            "ErrorUnit", EnumValue(RateErrorModel::ERROR_UNIT_PACKET));
    relay->SetAttribute("ReceiveErrorModel", PointerValue(em));
    relay->GetObject<PointToPointNetDevice> ()->SetAttribute ("DataRate", StringValue("3Mbps"));
    wbw << Simulator::Now().GetSeconds() << " "<< 0 << std::endl;
    }
}


void ChangeP2PAttributesPeriodically (NetDeviceContainer dish_sat, NetDeviceContainer sat_gs, Ptr<QuicL4Protocol> &tcp, Ptr<QuicSocketBase> &tcpsb, NodeContainer tcpsender, bool start) {
    
    // BindTcpSocket(tcpsender, tcp, tcpsb);
    Ptr<NetDevice> sender1;
    Ptr<NetDevice> sender2;
    if(upLoad)
    {
        sender1 = dish_sat.Get(0);
        sender2 = sat_gs.Get(0);
    }
    else
    {
        sender1 = dish_sat.Get(1);
        sender2 = sat_gs.Get(1);
    }
    
    std::random_device rd; // 获取随机数种子
    std::mt19937 gen(rd()); // 使用Mersenne Twister算法
    std::normal_distribution<> distrib(0.0, 0.05);

    // 增加计数器
    double now = Simulator::Now().GetSeconds() + starStart;
    std::cout<< "Now == " << now << std::endl;
    double next = std::ceil((now+0.001) / 0.5) * 0.5;
    std::cout<< "Next == " << next << std::endl;
    // 更改DataRate和Delay，示例中为每次不同
   
    
    std::vector<std::string> bw = ReadSpecificLineAndSplit(bw_filename, static_cast<int>(std::floor(now))%5731);
    std::vector<std::string> latency = ReadSpecificLineAndSplit(latency_filename, static_cast<int>(std::floor(now))%5731);
    uint8_t linkPort;
    uint8_t linkYue;
    if(upLoad)
    {
        linkPort = 1;
        linkYue = 3;
    }
    else
    {
        linkPort = 3;
        linkYue = 6;
    }
    double bw1 = roundToDecimal(std::stod(bw[linkPort])*(1+distrib(gen))/linkYue,2);
    double bw2 = roundToDecimal(std::stod(bw[linkPort+1])*(1+distrib(gen))/linkYue/satSerUser,2);
    double delay1 = roundToDecimal(std::stod(latency[1])*1000,2);
    double delay2 = roundToDecimal(std::stod(latency[2])*1000,2);
    int switch_flag = std::stoi(latency[3]);

    if(isRain && now >= rainTimeStart && now <= rainTimeStop)
    {
        // 降雨
        float bw_rain = (45 + rand()%31) / 100.0;       // 降雨带宽约束
        float loss_rain = (100 + rand()%201) / 100.0;   // 降雨丢包率增加
        bw1 = bw1 * bw_rain;
        bw2 = bw2 * bw_rain;
        float rainLoss = error_rate * loss_rain;
        std::cout << "rain    " << "bw_rain = " << bw_rain << std::endl;

        Ptr<RateErrorModel> em = CreateObjectWithAttributes<RateErrorModel>("ErrorRate", DoubleValue(rainLoss), 
                            "ErrorUnit", EnumValue(RateErrorModel::ERROR_UNIT_PACKET));
        sat_gs.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(em));
    }
    else
    {
        Ptr<RateErrorModel> em = CreateObjectWithAttributes<RateErrorModel>("ErrorRate", DoubleValue(error_rate), 
                            "ErrorUnit", EnumValue(RateErrorModel::ERROR_UNIT_PACKET));
        sat_gs.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(em));
    }

    if (switch_flag == 1 && start == false)
    {
        Simulator::Schedule (Seconds (0.05), &PhysicalLayerHandover, sat_gs, true);
        Simulator::Schedule (Seconds (0.05+5*(delay1+delay2)/100), &PhysicalLayerHandover, sat_gs, false);
        Simulator::Schedule (Seconds (next - now), &ChangeP2PAttributesPeriodically, dish_sat, sat_gs, tcp, tcpsb, tcpsender, true);
    }else
    {
        wbw << now - starStart << " "<< std::min(bw1, bw2) << std::endl;
        sender1->GetObject<PointToPointNetDevice> ()->SetAttribute ("DataRate", StringValue(std::to_string(bw1)+"Mbps"));
        sender2->GetObject<PointToPointNetDevice> ()->SetAttribute ("DataRate", StringValue(std::to_string(bw2)+"Mbps"));

        Ptr<PointToPointChannel> p2pChannel1 = sender1->GetChannel ()->GetObject<PointToPointChannel> ();
        p2pChannel1->SetAttribute ("Delay", StringValue(std::to_string(delay1)+"ms"));

        Ptr<PointToPointChannel> p2pChannel2 = sender1->GetChannel ()->GetObject<PointToPointChannel> ();
        p2pChannel2->SetAttribute ("Delay", StringValue(std::to_string(delay2)+"ms"));

        std::cout<< "Time: " << Simulator::Now ().GetSeconds () << "s, DataRate: " << std::to_string(bw1)+"Mbps " << std::to_string(bw2)+"Mbps" 
                                << ", Delay: " << std::to_string(delay1)+"ms " << std::to_string(delay2)+"ms" <<  std::endl;

        Simulator::Schedule (Seconds (next - now), &ChangeP2PAttributesPeriodically, dish_sat, sat_gs, tcp, tcpsb, tcpsender, false);
    }
    
}


int
main(int argc, char* argv[])
{
    // Naming the output directory using local system time
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%d-%m-%Y-%I-%M-%S", timeinfo);
    std::string currentTime(buffer);
    std::string tcpTypeId = "QuicVegas";
    std::string queueDisc = "FifoQueueDisc";
    uint32_t delAckCount = 2;
    bool bql = true;
    bool enablePcap = false;
    Time stopTime = Seconds(100);
    queueDisc = std::string("ns3::") + queueDisc;
    uint32_t kMaxTLPs = 2;


    SeedManager::SetSeed (30);
    // SeedManager::SetRun (0);

    Config::SetDefault("ns3::QuicL4Protocol::SocketType", StringValue("ns3::" + tcpTypeId));

    // The maximum send buffer size is set to 4194304 bytes (4MB) and the
    // maximum receive buffer size is set to 6291456 bytes (6MB) in the Linux
    // kernel. The same buffer sizes are used as default in this example.
    // Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1 << 23));
    // Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 23));
    // Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    // Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(delAckCount));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(mtu_bytes));
    // Config::SetDefault("ns3::TcpSocketState::EnablePacing", BooleanValue(true));
    // Config::SetDefault("ns3::TcpSocketState::MaxPacingRate", DataRateValue(DataRate("4Gbps")));
    Config::SetDefault("ns3::DropTailQueue<Packet>::MaxSize", QueueSizeValue(QueueSize("1p")));
    // Config::SetDefault ("ns3::QuicSocketBase::LegacyCongestionControl", BooleanValue(true));
    Config::SetDefault(queueDisc + "::MaxSize", QueueSizeValue(QueueSize(satQueue)));
    // Config::SetDefault("ns3::QuicSocketBase::kMaxTLPs", UintegerValue(kMaxTLPs));
    // Config::SetDefault("ns3::QuicL4Protocol::sLockTime", TimeValue (MilliSeconds (10)));
    // Config::SetDefault("ns3::QuicBbr::RttWindowLength", TimeValue (Seconds (50)));
    // Config::SetDefault("ns3::QuicBbr::BwWindowLength", UintegerValue (500));

    Config::SetDefault ("ns3::QuicSocketBase::SocketRcvBufSize", UintegerValue (1 << bufferi));
    Config::SetDefault ("ns3::QuicSocketBase::SocketSndBufSize", UintegerValue (1 << bufferi));
    Config::SetDefault ("ns3::QuicStreamBase::StreamSndBufSize", UintegerValue (1 << bufferi));
    Config::SetDefault ("ns3::QuicStreamBase::StreamRcvBufSize", UintegerValue (1 << bufferi));

    NodeContainer dishes;
    NodeContainer leosat;
    NodeContainer gs;
    NodeContainer pops;
    uint32_t users = 1; //以后扩展为多流场景
    dishes.Create(users);
    leosat.Create(1);
    gs.Create(1);
    pops.Create(1);

    PointToPointHelper dishes_leosat_link;
    dishes_leosat_link.SetDeviceAttribute("DataRate", StringValue("2Mbps"));
    dishes_leosat_link.SetChannelAttribute("Delay", StringValue("2ms"));

    PointToPointHelper leosat_gs_link;
    leosat_gs_link.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    leosat_gs_link.SetChannelAttribute("Delay", StringValue("2ms"));
    
    PointToPointHelper gs_pops_link;
    gs_pops_link.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    gs_pops_link.SetChannelAttribute("Delay", StringValue("5ms"));


    NetDeviceContainer dishes_leosat = dishes_leosat_link.Install(dishes.Get(0), leosat.Get(0));
    NetDeviceContainer leosat_gs = leosat_gs_link.Install(leosat.Get(0), gs.Get(0));
    NetDeviceContainer gs_pops = gs_pops_link.Install(gs.Get(0), pops.Get(0));

    // Create the point-to-point link helpers
    // PointToPointHelper bottleneckLink;
    // bottleneckLink.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    // bottleneckLink.SetChannelAttribute("Delay", StringValue("10ms"));



    Ptr<RateErrorModel> em = CreateObjectWithAttributes<RateErrorModel>("ErrorRate", DoubleValue(error_rate), 
                            "ErrorUnit", EnumValue(RateErrorModel::ERROR_UNIT_PACKET)); // 设置误码率
    // dishes_leosat.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(em));
    // dishes_leosat.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em));
    leosat_gs.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em));


    // Install Stack
    // InternetStackHelper internet;
    QuicHelper internet;

    internet.InstallQuic(dishes);
    internet.InstallQuic(leosat);
    internet.InstallQuic(gs);
    internet.InstallQuic(pops);


    // Configure the root queue discipline
    TrafficControlHelper tch;
    tch.SetRootQueueDisc(queueDisc);

    if (bql)
    {
        tch.SetQueueLimits("ns3::DynamicQueueLimits", "HoldTime", StringValue("1000ms"));
    }

    QueueDiscContainer qd_leo;
    qd_leo = tch.Install(leosat.Get(0)->GetDevice(1));
    Simulator::ScheduleNow(&CheckQueueSize, qd_leo.Get(0));

    tch.Install(dishes_leosat);
    tch.Install(gs_pops);

    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.0");

    Ipv4InterfaceContainer ip_dish_leosat = ipv4.Assign(dishes_leosat);
    ipv4.NewNetwork();
    Ipv4InterfaceContainer ip_leosat_gs = ipv4.Assign(leosat_gs);
    ipv4.NewNetwork();
    Ipv4InterfaceContainer ip_gs_pops = ipv4.Assign(gs_pops);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    uint16_t port = 50001;

    Ipv4Address serIP;
    if(upLoad)
    {
        serIP = ip_gs_pops.GetAddress(1);
    }
    else
    {
        serIP = ip_dish_leosat.GetAddress(0);
    }

    // Install application on the sender
    BulkSendHelper source("ns3::QuicSocketFactory", InetSocketAddress(serIP, port));
    source.SetAttribute("MaxBytes", UintegerValue(file_size));
    ApplicationContainer sourceApps;
    if(upLoad)
    {
        sourceApps = source.Install(dishes.Get(0));
    }
    else
    {
        sourceApps = source.Install(pops.Get(0));
    }
    sourceApps.Start(Seconds(0.1));

    Ptr<QuicL4Protocol> tcp = CreateObject<QuicL4Protocol>();
    Ptr<QuicSocketBase> tcpsb = CreateObject<QuicSocketBase>();
    // Hook trace source after application starts

    uint8_t sendId;
    if(upLoad)
    {
        sendId = 0;
    }
    else
    {
        sendId = 3;
    }
    Simulator::Schedule(Seconds(0.1) + MilliSeconds(1), &TraceCwnd, sendId, 0);
    Simulator::Schedule(Seconds(0.1) + MilliSeconds(1), &TraceRTT, sendId, 0);
    sourceApps.Stop(stopTime);

    // Install application on the receiver
    PacketSinkHelper sink("ns3::QuicSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps;
    if(upLoad)
    {
        sinkApps = sink.Install(pops.Get(0));
    }
    else
    {
        sinkApps = sink.Install(dishes.Get(0));
    }
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(stopTime);


    // Create a new directory to store the output of the program
    dir = "results/" + tcpTypeId + '-' + std::to_string(static_cast<int>(isSeamless)) + '-' + currentTime + "/";
    MakeDirectories(dir);

    // Generate PCAP traces if it is enabled
    if (enablePcap)
    {
        MakeDirectories(dir + "pcap/");

        dishes_leosat_link.EnablePcapAll(dir + "/pcap/bbr", true);
        leosat_gs_link.EnablePcapAll(dir + "/pcap/bbr", true);
        gs_pops_link.EnablePcapAll(dir + "/pcap/bbr", true);
    }

    // Open files for writing throughput traces and queue size
    throughput.open(dir + "/throughput.dat", std::ios::out);
    queueSize.open(dir + "/queueSize.dat", std::ios::out);
    wbw.open(dir + "/realbw.dat", std::ios::out);

    NS_ASSERT_MSG(throughput.is_open(), "Throughput file was not opened correctly");
    NS_ASSERT_MSG(queueSize.is_open(), "Queue size file was not opened correctly");

    // Check for dropped packets using Flow Monitor
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Schedule(Seconds(0.1) + MilliSeconds(1), &TraceThroughput, monitor);
    if(upLoad)
    {
        Simulator::Schedule (Seconds (0.1)+ MilliSeconds(1), &ChangeP2PAttributesPeriodically, 
                            dishes_leosat, leosat_gs, tcp, tcpsb, dishes, false);
    }
    else
    {
        Simulator::Schedule (Seconds (0.1)+ MilliSeconds(1), &ChangeP2PAttributesPeriodically, 
                            dishes_leosat, leosat_gs, tcp, tcpsb, pops, false);
    }

    Simulator::Stop(stopTime + TimeStep(1));
    Simulator::Run();


    Simulator::Destroy();

    throughput.close();
    queueSize.close();

    return 0;
}
