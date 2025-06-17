/*
 * Copyright (c) 2018 Technische Universität Berlin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz@tkn.tu-berlin.de>
 * Modify: Muyuan Shen <muyuan_shen@hust.edu.cn>
 */

#include "quic-rl.h"

#include "quic-rl-env.h"

#include "ns3/core-module.h"
#include "ns3/log.h"
#include "ns3/node-list.h"
#include "ns3/object.h"
#include "ns3/simulator.h"
#include "ns3/quic-header.h"
#include "ns3/quic-l4-protocol.h"
#include "ns3/quic-socket-base.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("quic-rl-gym");

NS_OBJECT_ENSURE_REGISTERED(QuicSocketDerived);

TypeId
QuicSocketDerived::GetTypeId()
{
    static TypeId tid = TypeId("ns3::QuicSocketDerived")
                            .SetParent<QuicSocketBase>()
                            .SetGroupName("Internet")
                            .AddConstructor<QuicSocketDerived>();
    return tid;
}

TypeId
QuicSocketDerived::GetInstanceTypeId() const
{
    return QuicSocketDerived::GetTypeId();
}

QuicSocketDerived::QuicSocketDerived()
{
}

Ptr<TcpCongestionOps>
QuicSocketDerived::GetCongestionControlAlgorithm()
{
    return m_congestionControl;
}

QuicSocketDerived::~QuicSocketDerived()
{
}

NS_OBJECT_ENSURE_REGISTERED(QuicRlBase);

TypeId
QuicRlBase::GetTypeId()
{
    static TypeId tid = TypeId("ns3::QuicRlBase")
                            .SetParent<QuicCongestionOps>()
                            .SetGroupName("Internet")
                            .AddConstructor<QuicRlBase>();
    return tid;
}

QuicRlBase::QuicRlBase()
    : QuicCongestionOps()
{
    NS_LOG_FUNCTION(this);
    m_QuicSocket = nullptr;
    m_QuicEnvBase = nullptr;
}

QuicRlBase::QuicRlBase(const QuicRlBase& sock)
    : QuicCongestionOps(sock)
{
    NS_LOG_FUNCTION(this);
    m_QuicSocket = nullptr;
    m_QuicEnvBase = nullptr;
}

QuicRlBase::~QuicRlBase()
{
    m_QuicSocket = nullptr;
    m_QuicEnvBase = nullptr;
}

uint64_t
QuicRlBase::GenerateUuid()
{
    static uint64_t uuid = 0;
    uuid++;
    return uuid;
}

void
QuicRlBase::CreateGymEnv()
{
    NS_LOG_FUNCTION(this);
    // should never be called, only child classes: QuicRlEventBased and QuicRlTimeBased
}

void
QuicRlBase::ConnectSocketCallbacks()
{
    NS_LOG_FUNCTION(this);

    bool foundSocket = false;
    for (NodeList::Iterator i = NodeList::Begin(); i != NodeList::End(); ++i)
    {
        Ptr<Node> node = *i;
        Ptr<QuicL4Protocol> quic = node->GetObject<QuicL4Protocol>();

        ObjectVectorValue socketVec;
        quic->GetAttribute("SocketList", socketVec);
        NS_LOG_DEBUG("Node: " << node->GetId() << " QUIC socket num: " << socketVec.GetN());
        std::cout << "Node: " << node->GetId() << " QUIC socket num: " << socketVec.GetN() << std::endl;

        uint32_t sockNum = socketVec.GetN();

        for (uint32_t j = 0; j < sockNum; j++)
        {
            Ptr<Object> sockObj = socketVec.Get(j);
            
            Ptr<QuicUdpBinding> quicbinding = DynamicCast<QuicUdpBinding>(sockObj);
            Ptr<QuicSocketBase> quicSocket = quicbinding->m_quicSocket;
            NS_LOG_DEBUG("Node: " << node->GetId() << " QUIC Socket: " << quicSocket);
            if (!quicSocket)
            {
                std::cout << "Quic Socket Not Found" << std::endl;
                continue;
            }

            // std::cout << "Quic Socket Found" << std::endl;
            Ptr<QuicSocketDerived> dquicSocket = StaticCast<QuicSocketDerived>(quicSocket);
            Ptr<TcpCongestionOps> ca = dquicSocket->GetCongestionControlAlgorithm();
            NS_LOG_DEBUG("CA name: " << ca->GetName());
            Ptr<QuicRlBase> rlCa = DynamicCast<QuicRlBase>(ca);
            if (rlCa == this)
            {
                std::cout << "CA name: " << rlCa->GetName() << std::endl;
                NS_LOG_DEBUG("Found QuicRl CA!");
                foundSocket = true;
                m_QuicSocket = quicSocket;
                break;
            }
        }

        if (foundSocket)
        {
            break;
        }
    }

    NS_ASSERT_MSG(m_QuicSocket, "QUIC socket was not found.");

    if (m_QuicSocket)
    {
        NS_LOG_DEBUG("Found QUIC Socket: " << m_QuicSocket);
        m_QuicSocket->TraceConnectWithoutContext(
            "Tx",
            MakeCallback(&QuicEnvBase::TxPktTrace, m_QuicEnvBase));
        m_QuicSocket->TraceConnectWithoutContext(
            "Rx",
            MakeCallback(&QuicEnvBase::RxPktTrace, m_QuicEnvBase));
        NS_LOG_DEBUG("Connect socket callbacks " << m_QuicSocket->GetNode()->GetId());
        m_QuicEnvBase->SetNodeId(m_QuicSocket->GetNode()->GetId());
    }
}

std::string
QuicRlBase::GetName() const
{
    return "QuicRlBase";
}

void
QuicRlBase::OnAckReceived (Ptr<TcpSocketState> tcb,
                                  QuicSubheader &ack,
                                  std::vector<Ptr<QuicSocketTxItem> > newAcks,
                                  const struct RateSample *rs)
{
  NS_LOG_FUNCTION (this << rs);

  Ptr<QuicSocketState> tcbd = dynamic_cast<QuicSocketState*> (&(*tcb));
  NS_ASSERT_MSG (tcbd != nullptr, "tcb is not a QuicSocketState");

  tcbd->m_largestAckedPacket = SequenceNumber32 (
    ack.GetLargestAcknowledged ());

  // newAcks are ordered from the highest packet number to the smalles
  Ptr<QuicSocketTxItem> lastAcked = newAcks.at (0);

  NS_LOG_LOGIC ("Updating RTT estimate");
  // If the largest acked is newly acked, update the RTT.
  if (lastAcked->m_packetNumber == tcbd->m_largestAckedPacket)
    {
      tcbd->m_lastRtt = Now () - lastAcked->m_lastSent;
      UpdateRtt (tcbd, tcbd->m_lastRtt, Time (ack.GetAckDelay ()));
    }

  if ((tcbd->m_congState == TcpSocketState::CA_RECOVERY or
       tcbd->m_congState == TcpSocketState::CA_LOSS) and
       tcbd->m_endOfRecovery <= tcbd->m_largestAckedPacket)
    {
      tcbd->m_congState = TcpSocketState::CA_OPEN;
      CongestionStateSet (tcb, TcpSocketState::CA_OPEN);
      CwndEvent (tcb, TcpSocketState::CA_EVENT_COMPLETE_CWR);
    }

  NS_LOG_LOGIC ("Processing acknowledged packets");
  // Process each acked packet
  for (auto it = newAcks.rbegin (); it != newAcks.rend (); ++it)
    {
      if ((*it)->m_acked)
        {
          OnPacketAcked (tcb, (*it));
        }
    }
}

void
QuicRlBase::OnPacketSent (Ptr<TcpSocketState> tcb,
                                 SequenceNumber32 packetNumber,
                                 bool isAckOnly)
{
  NS_LOG_FUNCTION (this << packetNumber << isAckOnly);
  Ptr<QuicSocketState> tcbd = dynamic_cast<QuicSocketState*> (&(*tcb));
  NS_ASSERT_MSG (tcbd != 0, "tcb is not a QuicSocketState");

  tcbd->m_timeOfLastSentPacket = Now ();
  tcbd->m_highTxMark = packetNumber;
}

void
QuicRlBase::OnPacketAcked (Ptr<TcpSocketState> tcb,
                                  Ptr<QuicSocketTxItem> ackedPacket)
{
  NS_LOG_FUNCTION (this);
  Ptr<QuicSocketState> tcbd = dynamic_cast<QuicSocketState*> (&(*tcb));
  NS_ASSERT_MSG (tcbd != 0, "tcb is not a QuicSocketState");

//   OnPacketAckedCC (tcbd, ackedPacket);
  PktsAcked(tcb, tcb->m_segmentSize, tcb->m_lastRtt);
  IncreaseWindow(tcb, tcb->m_segmentSize);

  NS_LOG_LOGIC ("Handle possible RTO");
  // 带改
  // If a packet sent prior to RTO was acked, then the RTO  was spurious. Otherwise, inform congestion control.
  if (tcbd->m_rtoCount > 0
      and ackedPacket->m_packetNumber > tcbd->m_largestSentBeforeRto)
    {
      OnRetransmissionTimeoutVerified (tcb);
    }
  tcbd->m_handshakeCount = 0;
  tcbd->m_tlpCount = 0;
  tcbd->m_rtoCount = 0;
}

void
QuicRlBase::OnRetransmissionTimeoutVerified (
  Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this);
  Ptr<QuicSocketState> tcbd = dynamic_cast<QuicSocketState*> (&(*tcb));
  NS_ASSERT_MSG (tcbd != 0, "tcb is not a QuicSocketState");
  NS_LOG_INFO ("Loss state");
//   tcbd->m_cWnd = tcbd->m_kMinimumWindow;
  tcbd->m_congState = TcpSocketState::CA_LOSS;
  CongestionStateSet (tcb, TcpSocketState::CA_LOSS);
}

void
QuicRlBase::OnPacketsLost (
  Ptr<TcpSocketState> tcb, std::vector<Ptr<QuicSocketTxItem> > lostPackets)
{
  NS_LOG_LOGIC (this);
  Ptr<QuicSocketState> tcbd = dynamic_cast<QuicSocketState*> (&(*tcb));
  NS_ASSERT_MSG (tcbd != 0, "tcb is not a QuicSocketState");

  auto largestLostPacket = *(lostPackets.end () - 1);

  NS_LOG_INFO ("Go in recovery mode");
  // Start a new recovery epoch if the lost packet is larger than the end of the previous recovery epoch.
  if (!InRecovery (tcbd, largestLostPacket->m_packetNumber))
    {
    //   tcbd->m_endOfRecovery = tcbd->m_highTxMark;
    //   tcbd->m_cWnd *= tcbd->m_kLossReductionFactor;
    //   if (tcbd->m_cWnd < tcbd->m_kMinimumWindow)
    //     {
    //       tcbd->m_cWnd = tcbd->m_kMinimumWindow;
    //     }
    //   tcbd->m_ssThresh = tcbd->m_cWnd;

        tcbd->m_endOfRecovery = tcbd->m_highTxMark;
        tcbd->m_congState = TcpSocketState::CA_RECOVERY;
        CongestionStateSet (tcbd, TcpSocketState::CA_RECOVERY);
        tcbd->m_ssThresh = GetSsThresh(tcb, tcbd->m_bytesInFlight);
        // m_QuicEnvBase->loss_rate = lostPackets.size()/(tcb->m_lastAckedSackedBytes / tcb->m_segmentSize);
    }
}

uint32_t
QuicRlBase::GetSsThresh(Ptr<const TcpSocketState> state, uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << state << bytesInFlight);

    if (!m_QuicEnvBase)
    {
        CreateGymEnv();
    }

    uint32_t newSsThresh = 0;
    if (m_QuicEnvBase)
    {
        newSsThresh = m_QuicEnvBase->GetSsThresh(state, bytesInFlight);
    }

    return newSsThresh;
}

void
QuicRlBase::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (!m_QuicEnvBase)
    {
        CreateGymEnv();
    }

    if (m_QuicEnvBase)
    {
        m_QuicEnvBase->IncreaseWindow(tcb, segmentsAcked);
    }
}

void
QuicRlBase::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this);

    if (!m_QuicEnvBase)
    {
        CreateGymEnv();
    }

    if (m_QuicEnvBase)
    {
        m_QuicEnvBase->PktsAcked(tcb, segmentsAcked, rtt);
    }
}

void
QuicRlBase::CongestionStateSet(Ptr<TcpSocketState> tcb,
                              const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this);

    if (!m_QuicEnvBase)
    {
        CreateGymEnv();
    }

    if (m_QuicEnvBase)
    {
        m_QuicEnvBase->CongestionStateSet(tcb, newState);
    }
}

void
QuicRlBase::CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this);

    if (!m_QuicEnvBase)
    {
        CreateGymEnv();
    }

    if (m_QuicEnvBase)
    {
        m_QuicEnvBase->CwndEvent(tcb, event);
    }
}

Ptr<TcpCongestionOps>
QuicRlBase::Fork()
{
    return CopyObject<QuicRlBase>(this);
}

NS_OBJECT_ENSURE_REGISTERED(QuicRlTimeBased);

TypeId
QuicRlTimeBased::GetTypeId()
{
    static TypeId tid = TypeId("ns3::QuicRlTimeBased")
                            .SetParent<QuicRlBase>()
                            .SetGroupName("Internet")
                            .AddConstructor<QuicRlTimeBased>();
    return tid;
}

QuicRlTimeBased::QuicRlTimeBased()
    : QuicRlBase()
{
    NS_LOG_FUNCTION(this);
}

QuicRlTimeBased::QuicRlTimeBased(const QuicRlTimeBased& sock)
    : QuicRlBase(sock)
{
    NS_LOG_FUNCTION(this);
}

QuicRlTimeBased::~QuicRlTimeBased()
{
}

std::string
QuicRlTimeBased::GetName() const
{
    return "QuicRlTimeBased";
}

void
QuicRlTimeBased::CreateGymEnv()
{
    NS_LOG_FUNCTION(this);
    Ptr<QuicTimeStepEnv> env = CreateObject<QuicTimeStepEnv>();
    env->SetSocketUuid(QuicRlBase::GenerateUuid());
    m_QuicEnvBase = env;

    ConnectSocketCallbacks();
}

NS_OBJECT_ENSURE_REGISTERED(QuicRlEventBased);

TypeId
QuicRlEventBased::GetTypeId()
{
    static TypeId tid = TypeId("ns3::QuicRlEventBased")
                            .SetParent<QuicRlBase>()
                            .SetGroupName("Internet")
                            .AddConstructor<QuicRlEventBased>()
                            .AddAttribute("Reward",
                                          "Reward when increasing congestion window.",
                                          DoubleValue(1.0),
                                          MakeDoubleAccessor(&QuicRlEventBased::m_reward),
                                          MakeDoubleChecker<double>())
                            .AddAttribute("Penalty",
                                          "Penalty after a loss event.",
                                          DoubleValue(-10.0),
                                          MakeDoubleAccessor(&QuicRlEventBased::m_penalty),
                                          MakeDoubleChecker<double>());
    return tid;
}

QuicRlEventBased::QuicRlEventBased()
    : QuicRlBase()
{
    NS_LOG_FUNCTION(this);
}

QuicRlEventBased::QuicRlEventBased(const QuicRlEventBased& sock)
    : QuicRlBase(sock)
{
    NS_LOG_FUNCTION(this);
}

QuicRlEventBased::~QuicRlEventBased()
{
}

std::string
QuicRlEventBased::GetName() const
{
    return "QuicRlEventBased";
}

void
QuicRlEventBased::CreateGymEnv()
{
    NS_LOG_FUNCTION(this);
    Ptr<QuicEventBasedEnv> env = CreateObject<QuicEventBasedEnv>();
    env->SetSocketUuid(QuicRlBase::GenerateUuid());
    env->SetReward(m_reward);
    env->SetPenalty(m_penalty);
    m_QuicEnvBase = env;

    ConnectSocketCallbacks();
}

/*
######## Our deep reinforcement learning based CC ########
*/

// TypeId
// TcpLstRl::GetTypeId()
// {
//     static TypeId tid = TypeId("ns3::TcpLstRl")
//                             .SetParent<QuicRlBase>()
//                             .SetGroupName("Internet")
//                             .AddConstructor<TcpLstRl>()
//                             .AddAttribute("Reward",
//                                           "Reward when increasing congestion window.",
//                                           DoubleValue(1.0),
//                                           MakeDoubleAccessor(&TcpLstRl::m_reward),
//                                           MakeDoubleChecker<double>())
//                             .AddAttribute("Penalty",
//                                           "Penalty after a loss event.",
//                                           DoubleValue(-10.0),
//                                           MakeDoubleAccessor(&TcpLstRl::m_penalty),
//                                           MakeDoubleChecker<double>());
//     return tid;
// }

// TcpLstRl::TcpLstRl()
//     : QuicRlBase()
// {
//     NS_LOG_FUNCTION(this);
// }

// TcpLstRl::TcpLstRl(const TcpLstRl& sock)
//     : QuicRlBase(sock)
// {
//     NS_LOG_FUNCTION(this);
// }

// TcpLstRl::~TcpLstRl()
// {
// }

// std::string
// TcpLstRl::GetName() const
// {
//     return "TcpLstRl";
// }

// void
// TcpLstRl::CreateGymEnv()
// {
//     NS_LOG_FUNCTION(this);
//     // Ptr<TcpLstRlEnv> env = CreateObject<TcpLstRlEnv>();
//     // env->SetSocketUuid(QuicRlBase::GenerateUuid());
//     // env->SetReward(m_reward);
//     // env->SetPenalty(m_penalty);
//     // m_QuicEnvBase = env;

//     ConnectSocketCallbacks();
// }



} // namespace ns3
