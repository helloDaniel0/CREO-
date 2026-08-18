#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

using namespace ns3;

namespace
{

class CreoHandoverHeader : public Header
{
  public:
    enum MessageType : uint8_t
    {
        NOTIFICATION = 1,
        ACK = 2
    };

    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::CreoHandoverHeader")
                                .SetParent<Header>()
                                .SetGroupName("Network")
                                .AddConstructor<CreoHandoverHeader>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    void SetMessageType(MessageType type)
    {
        m_type = type;
    }

    MessageType GetMessageType() const
    {
        return m_type;
    }

    void SetSequence(uint32_t sequence)
    {
        m_sequence = sequence;
    }

    uint32_t GetSequence() const
    {
        return m_sequence;
    }

    void SetHandoverTime(Time handoverTime)
    {
        m_handoverTimeNs = handoverTime.GetNanoSeconds();
    }

    Time GetHandoverTime() const
    {
        return NanoSeconds(m_handoverTimeNs);
    }

    bool IsValid() const
    {
        return m_magic == MAGIC && m_version == VERSION &&
               (m_type == NOTIFICATION || m_type == ACK);
    }

    uint32_t GetSerializedSize() const override
    {
        return 16;
    }

    void Serialize(Buffer::Iterator iterator) const override
    {
        iterator.WriteHtonU16(m_magic);
        iterator.WriteU8(m_version);
        iterator.WriteU8(static_cast<uint8_t>(m_type));
        iterator.WriteHtonU32(m_sequence);
        iterator.WriteHtonU64(m_handoverTimeNs);
    }

    uint32_t Deserialize(Buffer::Iterator iterator) override
    {
        m_magic = iterator.ReadNtohU16();
        m_version = iterator.ReadU8();
        m_type = static_cast<MessageType>(iterator.ReadU8());
        m_sequence = iterator.ReadNtohU32();
        m_handoverTimeNs = iterator.ReadNtohU64();
        return GetSerializedSize();
    }

    void Print(std::ostream& stream) const override
    {
        stream << "type=" << static_cast<uint32_t>(m_type) << " seq=" << m_sequence
               << " tHO=" << GetHandoverTime().GetSeconds() << "s";
    }

  private:
    static constexpr uint16_t MAGIC = 0x4352;
    static constexpr uint8_t VERSION = 1;

    uint16_t m_magic{MAGIC};
    uint8_t m_version{VERSION};
    MessageType m_type{NOTIFICATION};
    uint32_t m_sequence{0};
    uint64_t m_handoverTimeNs{0};
};

NS_OBJECT_ENSURE_REGISTERED(CreoHandoverHeader);

class HandoverNotifier : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::HandoverNotifier")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<HandoverNotifier>();
        return tid;
    }

    void Configure(Ipv4Address peer,
                   uint16_t peerPort,
                   uint16_t localPort,
                   Time handoverTime,
                   Time leadTime,
                   Time retryInterval,
                   uint32_t maxAttempts)
    {
        m_peer = InetSocketAddress(peer, peerPort);
        m_localPort = localPort;
        m_handoverTime = handoverTime;
        m_leadTime = leadTime;
        m_retryInterval = retryInterval;
        m_maxAttempts = std::max(1u, maxAttempts);
    }

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_localPort));
        m_socket->Connect(m_peer);
        m_socket->SetRecvCallback(MakeCallback(&HandoverNotifier::ReceiveAck, this));

        Time sendTime = std::max(Simulator::Now(), m_handoverTime - m_leadTime);
        m_sendEvent = Simulator::Schedule(sendTime - Simulator::Now(),
                                          &HandoverNotifier::SendNotification,
                                          this);
    }

    void StopApplication() override
    {
        Simulator::Cancel(m_sendEvent);
        Simulator::Cancel(m_retryEvent);
        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

    void SendNotification()
    {
        if (m_acknowledged || m_attempts >= m_maxAttempts)
        {
            return;
        }

        Ptr<Packet> packet = Create<Packet>();
        CreoHandoverHeader header;
        header.SetMessageType(CreoHandoverHeader::NOTIFICATION);
        header.SetSequence(m_sequence);
        header.SetHandoverTime(m_handoverTime);
        packet->AddHeader(header);
        m_socket->Send(packet);
        ++m_attempts;

        std::cout << Simulator::Now().GetSeconds() << "s notification seq=" << m_sequence
                  << " attempt=" << m_attempts << " tHO=" << m_handoverTime.GetSeconds()
                  << "s\n";

        if (m_attempts < m_maxAttempts)
        {
            m_retryEvent = Simulator::Schedule(m_retryInterval,
                                               &HandoverNotifier::SendNotification,
                                               this);
        }
    }

    void ReceiveAck(Ptr<Socket> socket)
    {
        while (Ptr<Packet> packet = socket->Recv())
        {
            if (packet->GetSize() < CreoHandoverHeader().GetSerializedSize())
            {
                continue;
            }

            CreoHandoverHeader header;
            packet->RemoveHeader(header);
            if (!header.IsValid() || header.GetMessageType() != CreoHandoverHeader::ACK ||
                header.GetSequence() != m_sequence)
            {
                continue;
            }

            m_acknowledged = true;
            Simulator::Cancel(m_retryEvent);
            std::cout << Simulator::Now().GetSeconds() << "s ACK seq=" << m_sequence << '\n';
        }
    }

    Ptr<Socket> m_socket;
    Address m_peer;
    uint16_t m_localPort{9001};
    Time m_handoverTime{Seconds(2)};
    Time m_leadTime{MilliSeconds(100)};
    Time m_retryInterval{MilliSeconds(50)};
    uint32_t m_maxAttempts{3};
    uint32_t m_attempts{0};
    uint32_t m_sequence{1};
    bool m_acknowledged{false};
    EventId m_sendEvent;
    EventId m_retryEvent;
};

NS_OBJECT_ENSURE_REGISTERED(HandoverNotifier);

class HandoverReceiver : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::HandoverReceiver")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<HandoverReceiver>();
        return tid;
    }

    void Configure(uint16_t port, Callback<void, uint32_t> handoverCallback)
    {
        m_port = port;
        m_handoverCallback = handoverCallback;
    }

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
        m_socket->SetRecvCallback(MakeCallback(&HandoverReceiver::ReceiveNotification, this));
    }

    void StopApplication() override
    {
        Simulator::Cancel(m_handoverEvent);
        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

    void ReceiveNotification(Ptr<Socket> socket)
    {
        Address sender;
        while (Ptr<Packet> packet = socket->RecvFrom(sender))
        {
            if (packet->GetSize() < CreoHandoverHeader().GetSerializedSize())
            {
                continue;
            }

            CreoHandoverHeader header;
            packet->RemoveHeader(header);
            if (!header.IsValid() ||
                header.GetMessageType() != CreoHandoverHeader::NOTIFICATION)
            {
                continue;
            }

            SendAck(sender, header);
            if (header.GetSequence() == m_lastSequence)
            {
                continue;
            }

            m_lastSequence = header.GetSequence();
            Time delay = std::max(Time(0), header.GetHandoverTime() - Simulator::Now());
            m_handoverEvent = Simulator::Schedule(delay,
                                                  &HandoverReceiver::RunHandoverCallback,
                                                  this,
                                                  header.GetSequence());
            std::cout << Simulator::Now().GetSeconds()
                      << "s accepted notification seq=" << header.GetSequence()
                      << " callback in " << delay.GetMilliSeconds() << "ms\n";
        }
    }

    void SendAck(const Address& sender, const CreoHandoverHeader& notification)
    {
        Ptr<Packet> packet = Create<Packet>();
        CreoHandoverHeader ack;
        ack.SetMessageType(CreoHandoverHeader::ACK);
        ack.SetSequence(notification.GetSequence());
        ack.SetHandoverTime(notification.GetHandoverTime());
        packet->AddHeader(ack);
        m_socket->SendTo(packet, 0, sender);
    }

    void RunHandoverCallback(uint32_t sequence)
    {
        if (!m_handoverCallback.IsNull())
        {
            m_handoverCallback(sequence);
        }
    }

    Ptr<Socket> m_socket;
    uint16_t m_port{9000};
    uint32_t m_lastSequence{0};
    Callback<void, uint32_t> m_handoverCallback;
    EventId m_handoverEvent;
};

NS_OBJECT_ENSURE_REGISTERED(HandoverReceiver);

void
HandoverBoundary(uint32_t sequence)
{
    std::cout << Simulator::Now().GetSeconds()
              << "s execute connected-to-handover transition seq=" << sequence << '\n';
}

} // namespace

int
main(int argc, char* argv[])
{
    double baseRttMs = 50.0;
    double handoverTimeSeconds = 2.0;
    double leadTimeMs = 100.0;
    uint32_t maxAttempts = 3;

    CommandLine cmd(__FILE__);
    cmd.AddValue("baseRttMs", "Round-trip propagation delay", baseRttMs);
    cmd.AddValue("handoverTime", "Absolute handover time in seconds", handoverTimeSeconds);
    cmd.AddValue("leadTimeMs", "Notification lead time", leadTimeMs);
    cmd.AddValue("maxAttempts", "Initial transmission plus retries", maxAttempts);
    cmd.Parse(argc, argv);

    NodeContainer nodes;
    nodes.Create(2);

    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    pointToPoint.SetChannelAttribute("Delay", TimeValue(MilliSeconds(baseRttMs / 2.0)));
    NetDeviceContainer devices = pointToPoint.Install(nodes);

    InternetStackHelper internet;
    internet.Install(nodes);

    Ipv4AddressHelper addresses;
    addresses.SetBase("10.1.0.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = addresses.Assign(devices);

    constexpr uint16_t receiverPort = 9000;
    constexpr uint16_t notifierPort = 9001;
    Time handoverTime = Seconds(handoverTimeSeconds);
    Time retryInterval = MilliSeconds(baseRttMs);

    Ptr<HandoverReceiver> receiver = CreateObject<HandoverReceiver>();
    receiver->Configure(receiverPort, MakeCallback(&HandoverBoundary));
    nodes.Get(1)->AddApplication(receiver);
    receiver->SetStartTime(Seconds(0));
    receiver->SetStopTime(handoverTime + Seconds(1));

    Ptr<HandoverNotifier> notifier = CreateObject<HandoverNotifier>();
    notifier->Configure(interfaces.GetAddress(1),
                        receiverPort,
                        notifierPort,
                        handoverTime,
                        MilliSeconds(leadTimeMs),
                        retryInterval,
                        maxAttempts);
    nodes.Get(0)->AddApplication(notifier);
    notifier->SetStartTime(Seconds(0));
    notifier->SetStopTime(handoverTime + Seconds(1));

    Simulator::Stop(handoverTime + Seconds(1));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
