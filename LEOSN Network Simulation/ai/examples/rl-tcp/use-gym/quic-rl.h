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

#ifndef NS3_QUIC_RL_H
#define NS3_QUIC_RL_H

#include "ns3/quic-congestion-ops.h"
#include "ns3/quic-socket-base.h"
#include <ns3/ai-module.h>

namespace ns3
{

class QuicSocketBase;
class Time;
class QuicEnvBase;


// tcp-socket-base主要负责TCP套接字的基本操作和管理，包括创建、绑定、监听、连接、发送和接收数据。
// 它定义了TCP连接的生命周期和基本的接口
// used to get pointer to Congestion Algorithm
class QuicSocketDerived : public QuicSocketBase
{
  public:
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;

    QuicSocketDerived();
    ~QuicSocketDerived() override;

    Ptr<TcpCongestionOps> GetCongestionControlAlgorithm();
};


// tcp-congestion-ops专注于TCP拥塞控制算法的实现。
// 它包含不同的拥塞控制策略，如慢启动、拥塞避免和快速恢复等
class QuicRlBase : public QuicCongestionOps
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    QuicRlBase();

    /**
     * \brief Copy constructor.
     * \param sock object to copy.
     */
    QuicRlBase(const QuicRlBase& sock);

    ~QuicRlBase() override;

    std::string GetName() const override;

    // uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    // void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    // void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;
    // void CongestionStateSet(Ptr<TcpSocketState> tcb,
    //                         const TcpSocketState::TcpCongState_t newState) override;
    // void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override;
    // Ptr<TcpCongestionOps> Fork() override;

    virtual void CongestionStateSet (Ptr<TcpSocketState> tcb,
                                   const TcpSocketState::TcpCongState_t newState);

    virtual void OnPacketSent (Ptr<TcpSocketState> tcb, SequenceNumber32 packetNumber, bool isAckOnly);
    virtual void OnAckReceived (Ptr<TcpSocketState> tcb, QuicSubheader &ack,
                                std::vector<Ptr<QuicSocketTxItem> > newAcks, const struct RateSample *rs);
    virtual void OnPacketsLost (Ptr<TcpSocketState> tcb, std::vector<Ptr<QuicSocketTxItem> > lostPackets);

    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;

    virtual void CwndEvent (Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCAEvent_t event);
    virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                  uint32_t bytesInFlight);
    virtual void IncreaseWindow (Ptr<TcpSocketState> tcb,
                                uint32_t segmentsAcked);
    Ptr<TcpCongestionOps> Fork ();

  protected:
    void OnPacketAcked (Ptr<TcpSocketState> tcb, Ptr<QuicSocketTxItem> ackedPacket);
    virtual void OnRetransmissionTimeoutVerified (Ptr<TcpSocketState> tcb);

    static uint64_t GenerateUuid();
    virtual void CreateGymEnv();
    void ConnectSocketCallbacks();

    // OpenGymEnv interface
    Ptr<QuicSocketBase> m_QuicSocket;
    Ptr<QuicEnvBase> m_QuicEnvBase;
};

class QuicRlTimeBased : public QuicRlBase
{
  public:
    static TypeId GetTypeId();

    QuicRlTimeBased();
    QuicRlTimeBased(const QuicRlTimeBased& sock);
    ~QuicRlTimeBased() override;

    std::string GetName() const override;

  private:
    void CreateGymEnv() override;
};

class QuicRlEventBased : public QuicRlBase
{
  public:
    static TypeId GetTypeId();

    QuicRlEventBased();
    QuicRlEventBased(const QuicRlEventBased& sock);
    ~QuicRlEventBased() override;

    std::string GetName() const override;

  private:
    void CreateGymEnv() override;
    // OpenGymEnv env
    float m_reward{1.0};
    float m_penalty{-100.0};
};

/* 
######## Our deep reinforcement learning based CC ########
*/


// class TcpLstRl : public TcpRlBase
// {
//   public:
//     static TypeId GetTypeId();

//     TcpLstRl();
//     TcpLstRl(const TcpLstRl& sock);
//     ~TcpLstRl() override;

//     std::string GetName() const override;

//   private:
//     void CreateGymEnv() override;
//     // OpenGymEnv env
//     float m_reward{1.0};
//     float m_penalty{-100.0};
// };

} // namespace ns3

#endif // NS3_TCP_RL_H
