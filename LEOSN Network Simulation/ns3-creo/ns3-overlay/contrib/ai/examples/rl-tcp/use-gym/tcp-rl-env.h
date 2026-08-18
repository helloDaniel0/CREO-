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

#ifndef NS3_TCP_RL_ENV_H
#define NS3_TCP_RL_ENV_H

#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-rate-ops.h"
#include "ns3/tcp-bbr.h"
#include <ns3/ai-module.h>
#include <vector>

namespace ns3
{

class Packet;
class TcpHeader;
class TcpSocketBase;
class Time;

class TcpEnvBase : public OpenGymEnv
{
  public:
    TcpEnvBase();
    ~TcpEnvBase() override;
    static TypeId GetTypeId();
    void DoDispose() override;

    void SetNodeId(uint32_t id);
    void SetSocketUuid(uint32_t id);

    std::string GetTcpCongStateName(const TcpSocketState::TcpCongState_t state);
    std::string GetTcpCAEventName(const TcpSocketState::TcpCAEvent_t event);

    // OpenGym interface
    Ptr<OpenGymSpace> GetActionSpace() override;
    bool GetGameOver() override;
    float GetReward() override;
    std::string GetExtraInfo() override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

    Ptr<OpenGymSpace> GetObservationSpace() override = 0;
    Ptr<OpenGymDataContainer> GetObservation() override = 0;

    // trace packets, e.g. for calculating inter tx/rx time
    virtual void TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) = 0;
    virtual void RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) = 0;

    // TCP congestion control interface
    
    virtual uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) = 0;
    virtual void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) = 0;
    // optional functions used to collect obs
    virtual void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) = 0;
    virtual void CongestionStateSet(Ptr<TcpSocketState> tcb,
                                    const TcpSocketState::TcpCongState_t newState) = 0;
    virtual void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) = 0;

    void UsePacing();

    typedef enum
    {
        GET_SS_THRESH = 0,
        INCREASE_WINDOW,
        PKTS_ACKED,
        CONGESTION_STATE_SET,
        CWND_EVENT,
    } CalledFunc_t;

  protected:
    uint32_t m_nodeId;
    uint32_t m_socketUuid;

    bool is_pacing = false;

    // state
    // obs has to be implemented in child class

    // reward
    float m_envReward;

    // extra info
    std::string m_info;

    // actions
    uint32_t m_new_ssThresh;
    uint32_t m_new_cWnd;
    uint32_t m_new_pacing_rate;
};

class TcpTimeStepEnv : public TcpEnvBase
{
  public:
    TcpTimeStepEnv();
    ~TcpTimeStepEnv() override;
    static TypeId GetTypeId();
    void DoDispose() override;

    // OpenGym interface
    Ptr<OpenGymSpace> GetObservationSpace() override;
    Ptr<OpenGymDataContainer> GetObservation() override;

    // trace packets, e.g. for calculating inter tx/rx time
    void TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) override;
    void RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) override;

    // TCP congestion control interface
    void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    // optional functions used to collect obs                                                     
    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;

    // Modified
    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;         
    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override;
    void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override;
    bool HasCongControl();
    void CongControl(Ptr<TcpSocketState> tcb,
                     const TcpRateOps::TcpRateConnection& rc,
                     const TcpRateOps::TcpRateSample& rs);
    void SetTimeStep(Time nextstep);

    // BBR Protected
    // void SaveCwnd(Ptr<const TcpSocketState> tcb); // new

  private:
    void ScheduleNextStateRead();
    Time CalculateNextStep();
    bool m_started{false};
    Time m_timeStep;
    bool m_adaptiveStep{true};
    Time m_stepTimeCap{MilliSeconds(30)};
    Time m_smoothedRtt{MicroSeconds(0)};
    // state
    Ptr<const TcpSocketState> m_tcb;
    uint32_t m_priorCwnd{0}; // new
    std::vector<uint32_t> m_bytesInFlight;
    std::vector<uint32_t> m_segmentsAcked;

    uint64_t m_rttSampleNum{0};
    Time m_rttSum{MicroSeconds(0.0)};

    Time m_lastPktTxTime{MicroSeconds(0.0)};
    Time m_lastPktRxTime{MicroSeconds(0.0)};
    uint64_t m_interTxTimeNum{0};
    Time m_interTxTimeSum{MicroSeconds(0.0)};
    uint64_t m_interRxTimeNum{0};
    Time m_interRxTimeSum{MicroSeconds(0.0)};

    bool m_enableBurstPacingSample{true};
    uint32_t m_miniWindowSegments{15};
    uint32_t m_miniWindowAcked{0};
    uint32_t m_miniWindowFirstAcked{0};
    Time m_miniWindowStart{MicroSeconds(0.0)};
    std::vector<uint64_t> m_capacitySamples;
    uint64_t m_lastCapacitySample{0};
    uint64_t m_lossEvents{0};
};

class TcpEventBasedEnv : public TcpEnvBase
{
  public:
    TcpEventBasedEnv();
    ~TcpEventBasedEnv() override;
    static TypeId GetTypeId();
    void DoDispose() override;

    void SetReward(float value);
    void SetPenalty(float value);

    // OpenGym interface
    Ptr<OpenGymSpace> GetObservationSpace() override;
    Ptr<OpenGymDataContainer> GetObservation() override;

    // trace packets, e.g. for calculating inter tx/rx time
    void TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) override;
    void RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) override;

    // TCP congestion control interface
    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    // optional functions used to collect obs
    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;
    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override;
    void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override;

    bool HasCongControl();
    void CongControl(Ptr<TcpSocketState> tcb,
                     const TcpRateOps::TcpRateConnection& rc,
                     const TcpRateOps::TcpRateSample& rs);

  private:
    // state
    CalledFunc_t m_calledFunc;
    Ptr<const TcpSocketState> m_tcb;
    uint32_t m_bytesInFlight;
    uint32_t m_segmentsAcked;
    Time m_rtt;
    TcpSocketState::TcpCAEvent_t m_event;

    // reward
    float m_reward;
    float m_penalty;
};

/*
######## Our deep reinforcement learning based CC framework ########
*/

// class TcpLstRlEnv : public TcpEnvBase
// {
//   public:
//     TcpLstRlEnv();
//     ~TcpLstRlEnv() override;
//     static TypeId GetTypeId();
//     void DoDispose() override;

//     // OpenGym interface
//     Ptr<OpenGymSpace> GetObservationSpace() override;
//     Ptr<OpenGymDataContainer> GetObservation() override;

//     // trace packets, e.g. for calculating inter tx/rx time
//     void TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) override;
//     void RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>) override;

//     // TCP congestion control interface
//     uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
//     void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
//     // optional functions used to collect obs
//     void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override;
//     void CongestionStateSet(Ptr<TcpSocketState> tcb,
//                             const TcpSocketState::TcpCongState_t newState) override;
//     void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override;

//   private:
//     //state

//     //reward
// };

} // namespace ns3

#endif // NS3_TCP_RL_ENV_H
