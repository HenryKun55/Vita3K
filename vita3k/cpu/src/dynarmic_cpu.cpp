// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include "cpu/common.h"
#include <cpu/impl/dynarmic_cpu.h>
#include <cpu/state.h>
#include <util/log.h>

#include <mem/ptr.h>

#include <dynarmic/frontend/A32/a32_ir_emitter.h>
#include <dynarmic/interface/A32/coprocessor.h>
#include <dynarmic/interface/exclusive_monitor.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <semaphore>
#include <thread>
#include <utility>
#include <optional>
#include <string>

class ArmDynarmicCP15 : public Dynarmic::A32::Coprocessor {
    uint32_t tpidruro;
    uint32_t sctlr;
    uint32_t dacr;

public:
    using CoprocReg = Dynarmic::A32::CoprocReg;

    explicit ArmDynarmicCP15()
        : tpidruro(0)
        , sctlr(0)
        , dacr(0) {
    }

    ~ArmDynarmicCP15() override = default;

    std::optional<Callback> CompileInternalOperation(bool two, unsigned opc1, CoprocReg CRd,
        CoprocReg CRn, CoprocReg CRm,
        unsigned opc2) override {
        return std::nullopt;
    }

    CallbackOrAccessOneWord CompileSendOneWord(bool two, unsigned opc1, CoprocReg CRn,
        CoprocReg CRm, unsigned opc2) override {
        // MCR p15, 0, Rt, c13, c0, 3 — write TPIDRURO
        if (CRn == CoprocReg::C13 && CRm == CoprocReg::C0 && opc1 == 0 && opc2 == 3) {
            return &tpidruro;
        }

        // MCR p15, 0, Rt, c1, c0, 0 — write SCTLR
        if (!two && CRn == CoprocReg::C1 && CRm == CoprocReg::C0 && opc1 == 0 && opc2 == 0) {
            return &sctlr;
        }

        // MCR p15, 0, Rt, c3, c0, 0 — write DACR
        if (!two && CRn == CoprocReg::C3 && CRm == CoprocReg::C0 && opc1 == 0 && opc2 == 0) {
            return &dacr;
        }

        LOG_WARN("Unhandled CP15 MCR: two={} opc1={} CRn={} CRm={} opc2={}", two, opc1, (int)CRn, (int)CRm, opc2);
        return CallbackOrAccessOneWord{};
    }

    CallbackOrAccessTwoWords CompileSendTwoWords(bool two, unsigned opc, CoprocReg CRm) override {
        return CallbackOrAccessTwoWords{};
    }

    CallbackOrAccessOneWord CompileGetOneWord(bool two, unsigned opc1, CoprocReg CRn, CoprocReg CRm,
        unsigned opc2) override {
        // MRC p15, 0, Rt, c13, c0, 3 — read TPIDRURO (thread-local storage)
        if (CRn == CoprocReg::C13 && CRm == CoprocReg::C0 && opc1 == 0 && opc2 == 3) {
            return &tpidruro;
        }

        // MRC p15, 0, Rt, c1, c0, 0 — read SCTLR
        if (!two && CRn == CoprocReg::C1 && CRm == CoprocReg::C0 && opc1 == 0 && opc2 == 0) {
            return &sctlr;
        }

        // MRC p15, 0, Rt, c3, c0, 0 — read DACR
        if (!two && CRn == CoprocReg::C3 && CRm == CoprocReg::C0 && opc1 == 0 && opc2 == 0) {
            return &dacr;
        }

        LOG_WARN("Unhandled CP15 MRC: two={} opc1={} CRn={} CRm={} opc2={}", two, opc1, (int)CRn, (int)CRm, opc2);
        return CallbackOrAccessOneWord{};
    }

    CallbackOrAccessTwoWords CompileGetTwoWords(bool two, unsigned opc, CoprocReg CRm) override {
        return CallbackOrAccessTwoWords{};
    }

    std::optional<Callback> CompileLoadWords(bool two, bool long_transfer, CoprocReg CRd,
        std::optional<std::uint8_t> option) override {
        return std::nullopt;
    }

    std::optional<Callback> CompileStoreWords(bool two, bool long_transfer, CoprocReg CRd,
        std::optional<std::uint8_t> option) override {
        return std::nullopt;
    }

    void set_tpidruro(uint32_t tpidruro) {
        this->tpidruro = tpidruro;
    }

    uint32_t get_tpidruro() const {
        return tpidruro;
    }
};

// PS Vita speed emulation. Set VITA3K_CPU_MHZ to the effective guest instruction rate in
// millions/sec per core (444 = Cortex-A9 overclocked at IPC 1; real IPC is lower). When set:
//  - each guest thread is slowed to that instruction rate,
//  - HLE work (native host code) is billed back as guest time via vita_speed_charge*,
//  - at most VITA3K_CORES (default 3, the cores a Vita app gets) guest threads run at once.
// 0/unset = off, no behaviour change.
static double throttle_ips() {
    static const double ips = [] {
        const char *env = std::getenv("VITA3K_CPU_MHZ");
        const double mhz = env ? std::atof(env) : 0.0;
        if (mhz > 0)
            LOG_INFO("CPU throttle enabled: {} M guest instructions/s per thread", mhz);
        return mhz > 0 ? mhz * 1e6 : 0.0;
    }();
    return ips;
}

// Guest instructions per JIT slice before control returns to AddTicks.
static constexpr uint64_t THROTTLE_SLICE = 20000;

static std::counting_semaphore<64> &core_slots() {
    static std::counting_semaphore<64> slots([] {
        const char *env = std::getenv("VITA3K_CORES");
        const int n = std::clamp(env ? std::atoi(env) : 3, 1, 64);
        LOG_INFO("Guest core limit: {}", n);
        return n;
    }());
    return slots;
}

// HLE time owed by the guest thread running on this host thread; paid at the next AddTicks.
static thread_local uint64_t hle_debt = 0;
// HLE time billed during the current import call (for the profiler).
static thread_local uint64_t hle_call_cost = 0;

// Guest sampling profiler for speed mode: VITA3K_PROFILE=<file>. Takes one sample every
// THROTTLE_SLICE guest instructions (the countdown carries across JIT exits, so the PC is
// wherever the countdown expired, unbiased by SVC exits), and records HLE time separately
// at the import stub PC with the caller's LR (vita_speed_profile_hle). Rewrites <file>
// every 5 s as "thread pc lr weight" lines. Symbolize against the app's ELF.
struct GuestProfiler {
    std::mutex mutex;
    std::unordered_map<uint64_t, uint64_t> samples; // key: thread << 32 | pc ; separate LR map
    std::unordered_map<uint64_t, uint32_t> lr_of;
    std::chrono::steady_clock::time_point last_dump = std::chrono::steady_clock::now();
    std::string path;

    void add(uint32_t thread, uint32_t pc, uint32_t lr, uint64_t weight) {
        const uint64_t key = (uint64_t(thread) << 32) | pc;
        std::lock_guard lock(mutex);
        samples[key] += weight;
        lr_of[key] = lr;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_dump > std::chrono::seconds(5)) {
            last_dump = now;
            std::ofstream out(path, std::ios::trunc);
            for (const auto &[k, w] : samples)
                out << (k >> 32) << ' ' << std::hex << (k & 0xFFFFFFFF) << ' ' << lr_of[k] << std::dec << ' ' << w << '\n';
        }
    }
};

static GuestProfiler *guest_profiler() {
    static GuestProfiler *prof = []() -> GuestProfiler * {
        const char *env = std::getenv("VITA3K_PROFILE");
        if (!env || !*env)
            return nullptr;
        auto *p = new GuestProfiler();
        p->path = env;
        LOG_INFO("Guest profiler writing to {}", p->path);
        return p;
    }();
    return prof;
}

bool vita_speed_enabled() {
    return throttle_ips() > 0;
}

void vita_speed_charge(uint64_t guest_instructions) {
    if (vita_speed_enabled()) {
        hle_debt += guest_instructions;
        hle_call_cost += guest_instructions;
    }
}

void vita_speed_charge_bytes(uint64_t bytes, double vita_mb_per_s) {
    vita_speed_charge(static_cast<uint64_t>(bytes * throttle_ips() / (vita_mb_per_s * 1e6)));
}

void vita_speed_profile_hle(uint32_t thread_id, uint32_t pc, uint32_t lr) {
    const uint64_t cost = std::exchange(hle_call_cost, 0);
    if (GuestProfiler *prof = guest_profiler(); prof && cost)
        prof->add(thread_id, pc, lr, cost);
}

class ArmDynarmicCallback : public Dynarmic::A32::UserCallbacks {
    friend class DynarmicCPU;

    CPUState *parent;
    DynarmicCPU *cpu;

    using Clock = std::chrono::steady_clock;
    Clock::time_point window_start = Clock::now();
    uint64_t window_ticks = 0;
    // Instructions left until the next profiler sample; persists across Run() calls.
    uint64_t slice_left = THROTTLE_SLICE;

public:
    explicit ArmDynarmicCallback(CPUState &parent, DynarmicCPU &cpu)
        : parent(&parent)
        , cpu(&cpu) {}

    ~ArmDynarmicCallback() override = default;

    std::optional<std::uint32_t> MemoryReadCode(Dynarmic::A32::VAddr addr) override {
        if (cpu->log_mem)
            LOG_TRACE("Instruction fetch at address 0x{:X}", addr);
        return MemoryRead32(addr);
    }

    static void TraceInstruction(uint64_t self_, uint64_t address, uint64_t is_thumb) {
        ArmDynarmicCallback &self = *reinterpret_cast<ArmDynarmicCallback *>(self_);

        std::string disassembly = [&]() -> std::string {
            if (!address || !Ptr<uint32_t>{ (uint32_t)address }.valid(*self.parent->mem)) {
                return "invalid address";
            }
            return disassemble(*self.parent, address);
        }();
        LOG_TRACE("{} ({}): {} {}", log_hex(self_), self.parent->thread_id, log_hex(address), disassembly);
    }

    void PreCodeTranslationHook(bool is_thumb, Dynarmic::A32::VAddr pc, Dynarmic::A32::IREmitter &ir) override {
        if (cpu->log_code) {
            ir.CallHostFunction(&TraceInstruction, ir.Imm64((uint64_t)this), ir.Imm64(pc), ir.Imm64(is_thumb));
        }
    }

    // Log guest thread id + words on the stack that look like return addresses into the
    // main executable, so a crash can be walked back to its caller with addr2line.
    void LogStackReturnAddresses() {
        const uint32_t sp = this->cpu->get_sp();
        std::string out;
        for (uint32_t off = 0; off < 32768; off += 4) {
            Ptr<uint32_t> w{ sp + off };
            if (!w.valid(*parent->mem))
                break;
            const uint32_t v = *w.get(*parent->mem);
            if ((v & 1) && v >= 0x81000000 && v < 0x81800000)
                out += fmt::format(" {:x}", v - 1);
        }
        LOG_ERROR("Guest thread {} stack return candidates:{}", parent->thread_id, out);
    }

    template <typename T>
    T MemoryRead(Dynarmic::A32::VAddr addr) {
        Ptr<T> ptr{ addr };
        if (!ptr || !ptr.valid(*parent->mem) || ptr.address() < parent->mem->host_page_size) {
            LOG_ERROR("Invalid read of uint{}_t at address: 0x{:x}\n{}", sizeof(T) * 8, addr, this->cpu->save_context().description());
            LogStackReturnAddresses();

            auto pc = this->cpu->get_pc();
            if (pc < parent->mem->host_page_size)
                LOG_CRITICAL("PC is 0x{:x}", pc);
            else
                LOG_ERROR("Executing: {}", disassemble(*parent, pc, nullptr));
            return 0;
        }

        T ret = *ptr.get(*parent->mem);
        if (cpu->log_mem) {
            LOG_TRACE("Read uint{}_t at address: 0x{:x}, val = 0x{:x}", sizeof(T) * 8, addr, ret);
        }
        return ret;
    }

    uint8_t MemoryRead8(Dynarmic::A32::VAddr addr) override {
        return MemoryRead<uint8_t>(addr);
    }

    uint16_t MemoryRead16(Dynarmic::A32::VAddr addr) override {
        return MemoryRead<uint16_t>(addr);
    }

    uint32_t MemoryRead32(Dynarmic::A32::VAddr addr) override {
        return MemoryRead<uint32_t>(addr);
    }

    uint64_t MemoryRead64(Dynarmic::A32::VAddr addr) override {
        return MemoryRead<uint64_t>(addr);
    }

    template <typename T>
    void MemoryWrite(Dynarmic::A32::VAddr addr, T value) {
        Ptr<T> ptr{ addr };
        if (!ptr || !ptr.valid(*parent->mem) || ptr.address() < parent->mem->host_page_size) {
            LOG_ERROR("Invalid write of uint{}_t at addr: 0x{:x}, val = 0x{:x}\n{}", sizeof(T) * 8, addr, value, this->cpu->save_context().description());
            LogStackReturnAddresses();

            auto pc = this->cpu->get_pc();
            if (pc < parent->mem->host_page_size)
                LOG_CRITICAL("PC is 0x{:x}", pc);
            else
                LOG_ERROR("Executing: {}", disassemble(*parent, pc, nullptr));
            return;
        }

        *ptr.get(*parent->mem) = value;
        if (cpu->log_mem) {
            LOG_TRACE("Write uint{}_t at addr: 0x{:x}, val = 0x{:x}", sizeof(T) * 8, addr, value);
        }
    }

    void MemoryWrite8(Dynarmic::A32::VAddr addr, uint8_t value) override {
        MemoryWrite<uint8_t>(addr, value);
    }

    void MemoryWrite16(Dynarmic::A32::VAddr addr, uint16_t value) override {
        MemoryWrite<uint16_t>(addr, value);
    }

    void MemoryWrite32(Dynarmic::A32::VAddr addr, uint32_t value) override {
        MemoryWrite<uint32_t>(addr, value);
    }

    void MemoryWrite64(Dynarmic::A32::VAddr addr, uint64_t value) override {
        MemoryWrite<uint64_t>(addr, value);
    }

    template <typename T>
    bool MemoryWriteExclusive(Dynarmic::A32::VAddr addr, T value, T expected) {
        Ptr<T> ptr{ addr };
        if (!ptr || !ptr.valid(*parent->mem) || ptr.address() < parent->mem->host_page_size) {
            LOG_ERROR("Invalid exclusive write of uint{}_t at addr: 0x{:x}, val = 0x{:x}, expected = 0x{:x}\n{}", sizeof(T) * 8, addr, value, expected, this->cpu->save_context().description());

            auto pc = this->cpu->get_pc();
            if (pc < parent->mem->host_page_size)
                LOG_CRITICAL("PC is 0x{:x}", pc);
            else
                LOG_ERROR("Executing: {}", disassemble(*parent, pc, nullptr));
            return false;
        }

        auto result = Ptr<T>(addr).atomic_compare_and_swap(*parent->mem, value, expected);
        if (cpu->log_mem) {
            LOG_TRACE("Write uint{}_t at addr: 0x{:x}, val = 0x{:x}, expected = 0x{:x}", sizeof(T) * 8, addr, value, expected);
        }
        return result;
    }

    bool MemoryWriteExclusive8(Dynarmic::A32::VAddr addr, uint8_t value, uint8_t expected) override {
        return MemoryWriteExclusive(addr, value, expected);
    }

    bool MemoryWriteExclusive16(Dynarmic::A32::VAddr addr, uint16_t value, uint16_t expected) override {
        return MemoryWriteExclusive(addr, value, expected);
    }

    bool MemoryWriteExclusive32(Dynarmic::A32::VAddr addr, uint32_t value, uint32_t expected) override {
        return MemoryWriteExclusive(addr, value, expected);
    }

    bool MemoryWriteExclusive64(Dynarmic::A32::VAddr addr, uint64_t value, uint64_t expected) override {
        return MemoryWriteExclusive(addr, value, expected); // Ptr<uint64_t>(addr).atomic_compare_and_swap(*parent->mem, value, expected);
    }

    void InterpreterFallback(Dynarmic::A32::VAddr addr, size_t num_insts) override {
        LOG_ERROR("Unimplemented instruction at address {}:\n{}", log_hex(addr), save_context(*parent).description());
    }

    void ExceptionRaised(uint32_t pc, Dynarmic::A32::Exception exception) override {
        switch (exception) {
        case Dynarmic::A32::Exception::Breakpoint: {
            cpu->break_ = true;
            cpu->jit->HaltExecution();
            if (cpu->is_thumb_mode())
                cpu->set_pc(pc | 1);
            else
                cpu->set_pc(pc);
            break;
        }
        case Dynarmic::A32::Exception::WaitForInterrupt: {
            cpu->halted = true;
            cpu->jit->HaltExecution();
            break;
        }
        case Dynarmic::A32::Exception::PreloadDataWithIntentToWrite:
        case Dynarmic::A32::Exception::PreloadData:
        case Dynarmic::A32::Exception::PreloadInstruction:
        case Dynarmic::A32::Exception::SendEvent:
        case Dynarmic::A32::Exception::SendEventLocal:
        case Dynarmic::A32::Exception::WaitForEvent:
            break;
        case Dynarmic::A32::Exception::Yield:
            break;
        case Dynarmic::A32::Exception::UndefinedInstruction:
            LOG_WARN("Undefined instruction at address 0x{:X}, instruction 0x{:X} ({})", pc, MemoryReadCode(pc).value(), disassemble(*parent, pc, nullptr));
            InterpreterFallback(pc, 1);
            break;
        case Dynarmic::A32::Exception::UnpredictableInstruction:
            LOG_WARN("Unpredictable instruction at address 0x{:X}, instruction 0x{:X} ({})", pc, MemoryReadCode(pc).value(), disassemble(*parent, pc, nullptr));
            InterpreterFallback(pc, 1);
            break;
        case Dynarmic::A32::Exception::DecodeError: {
            LOG_WARN("Decode error at address 0x{:X}, instruction 0x{:X} ({})", pc, MemoryReadCode(pc).value(), disassemble(*parent, pc, nullptr));
            InterpreterFallback(pc, 1);
            break;
        }
        default:
            LOG_WARN("Unknown exception {} Raised at pc = 0x{:x}", static_cast<size_t>(exception), pc);
            LOG_TRACE("at address 0x{:X}, instruction 0x{:X} ({})", pc, MemoryReadCode(pc).value(), disassemble(*parent, pc, nullptr));
        }
    }

    void CallSVC(uint32_t svc) override {
        parent->svc_called = true;
        parent->svc = svc;
        cpu->jit->HaltExecution(Dynarmic::HaltReason::UserDefined8);
    }

    void AddTicks(uint64_t ticks) override {
        const double ips = throttle_ips();
        if (ips <= 0)
            return;

        window_ticks += ticks + std::exchange(hle_debt, 0);
        if (ticks >= slice_left) {
            slice_left = THROTTLE_SLICE;
            if (GuestProfiler *prof = guest_profiler()) {
                const auto &regs = cpu->jit->Regs();
                prof->add(parent->thread_id, regs[15], regs[14], THROTTLE_SLICE);
            }
        } else {
            slice_left -= ticks;
        }
        const auto now = Clock::now();
        const auto elapsed = now - window_start;
        const auto expected = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(window_ticks / ips));

        if (expected > elapsed) {
            // Ran faster than a Vita would: pay the difference once it is worth a sleep.
            if (expected - elapsed >= std::chrono::milliseconds(1))
                std::this_thread::sleep_until(window_start + expected);
        } else if (elapsed - expected > std::chrono::milliseconds(2)) {
            // Thread was blocked/idle (or host was slower): don't bank credit for a burst later.
            window_start = now;
            window_ticks = 0;
        }
    }

    uint64_t GetTicksRemaining() override {
        return throttle_ips() > 0 ? slice_left : 1ull << 60;
    }
};

Dynarmic::ExclusiveMonitor DynarmicCPU::shared_monitor(MAX_CORE_COUNT);

std::unique_ptr<Dynarmic::A32::Jit> DynarmicCPU::make_jit() {
    Dynarmic::A32::UserConfig config{};
    config.arch_version = Dynarmic::A32::ArchVersion::v7;
    config.callbacks = cb.get();
    if (parent->mem->use_page_table) {
        config.page_table = (log_mem || !cpu_opt) ? nullptr : reinterpret_cast<decltype(config.page_table)>(parent->mem->page_table.get());
        config.absolute_offset_page_table = true;
    } else if (!log_mem && cpu_opt) {
        config.fastmem_pointer = std::bit_cast<uintptr_t>(parent->mem->memory.get());
    }
    config.hook_hint_instructions = true;
    config.global_monitor = &shared_monitor;
    config.coprocessors[15] = cp15;
    config.processor_id = core_id;
    config.optimizations = cpu_opt ? Dynarmic::all_safe_optimizations : Dynarmic::no_optimizations;
    config.enable_cycle_counting = throttle_ips() > 0;

    return std::make_unique<Dynarmic::A32::Jit>(config);
}

DynarmicCPU::DynarmicCPU(CPUState *state, std::size_t processor_id, bool cpu_opt)
    : parent(state)
    , cb(std::make_unique<ArmDynarmicCallback>(*state, *this))
    , cp15(std::make_shared<ArmDynarmicCP15>())
    , core_id(processor_id)
    , cpu_opt(cpu_opt) {
    jit = make_jit();
}

DynarmicCPU::~DynarmicCPU() = default;

int DynarmicCPU::run() {
    halted = false;
    break_ = false;
    parent->svc_called = false;
    Dynarmic::HaltReason halt_reason;
    const bool limit_cores = throttle_ips() > 0;
    if (limit_cores)
        core_slots().acquire();
    do {
        halt_reason = jit->Run();
    } while ((halt_reason == Dynarmic::HaltReason::Step) || (halt_reason == Dynarmic::HaltReason::CacheInvalidation));
    if (limit_cores)
        core_slots().release();

    return halted;
}

int DynarmicCPU::step() {
    parent->svc_called = false;
    jit->Step();
    return 0;
}

bool DynarmicCPU::hit_breakpoint() {
    return break_;
}

void DynarmicCPU::trigger_breakpoint() {
    break_ = true;
    stop();
}

void DynarmicCPU::set_log_code(bool log) {
    if (log_code == log)
        return;

    log_code = log;
    jit = make_jit();
}

void DynarmicCPU::set_log_mem(bool log) {
    if (log_mem == log)
        return;

    log_mem = log;
    jit = make_jit();
}

bool DynarmicCPU::get_log_code() {
    return log_code;
}

bool DynarmicCPU::get_log_mem() {
    return log_mem;
}

void DynarmicCPU::stop() {
    jit->HaltExecution();
}

uint32_t DynarmicCPU::get_reg(uint8_t idx) {
    return jit->Regs()[idx];
}

uint32_t DynarmicCPU::get_sp() {
    return jit->Regs()[13];
}

uint32_t DynarmicCPU::get_pc() {
    return jit->Regs()[15];
}

void DynarmicCPU::set_reg(uint8_t idx, uint32_t val) {
    jit->Regs()[idx] = val;
}

void DynarmicCPU::set_cpsr(uint32_t val) {
    jit->SetCpsr(val);
}

uint32_t DynarmicCPU::get_tpidruro() {
    return cp15->get_tpidruro();
}

void DynarmicCPU::set_tpidruro(uint32_t val) {
    cp15->set_tpidruro(val);
}

void DynarmicCPU::set_pc(uint32_t val) {
    if (val & 1) {
        set_cpsr(get_cpsr() | 0x20);
        val = val & 0xFFFFFFFE;
    } else {
        set_cpsr(get_cpsr() & 0xFFFFFFDF);
        val = val & 0xFFFFFFFC;
    }
    jit->Regs()[15] = val;
}

void DynarmicCPU::set_lr(uint32_t val) {
    jit->Regs()[14] = val;
}

void DynarmicCPU::set_sp(uint32_t val) {
    jit->Regs()[13] = val;
}

uint32_t DynarmicCPU::get_cpsr() {
    return jit->Cpsr();
}

uint32_t DynarmicCPU::get_fpscr() {
    return jit->Fpscr();
}

void DynarmicCPU::set_fpscr(uint32_t val) {
    jit->SetFpscr(val);
}

CPUContext DynarmicCPU::save_context() {
    CPUContext ctx;
    ctx.cpu_registers = jit->Regs();
    static_assert(sizeof(ctx.fpu_registers) == sizeof(jit->ExtRegs()));
    memcpy(ctx.fpu_registers.data(), jit->ExtRegs().data(), sizeof(ctx.fpu_registers));
    ctx.fpscr = jit->Fpscr();
    ctx.cpsr = jit->Cpsr();

    return ctx;
}

void DynarmicCPU::load_context(const CPUContext &ctx) {
    jit->Regs() = ctx.cpu_registers;
    static_assert(sizeof(ctx.fpu_registers) == sizeof(jit->ExtRegs()));
    memcpy(jit->ExtRegs().data(), ctx.fpu_registers.data(), sizeof(ctx.fpu_registers));
    jit->SetCpsr(ctx.cpsr);
    jit->SetFpscr(ctx.fpscr);
}

uint32_t DynarmicCPU::get_lr() {
    return jit->Regs()[14];
}

float DynarmicCPU::get_float_reg(uint8_t idx) {
    return std::bit_cast<float>(jit->ExtRegs()[idx]);
}

void DynarmicCPU::set_float_reg(uint8_t idx, float val) {
    jit->ExtRegs()[idx] = std::bit_cast<uint32_t>(val);
}

bool DynarmicCPU::is_thumb_mode() {
    return jit->Cpsr() & 0x20;
}

std::size_t DynarmicCPU::processor_id() const {
    return core_id;
}

void DynarmicCPU::invalidate_jit_cache(Address start, size_t length) {
    jit->InvalidateCacheRange(start, length);
}

void DynarmicCPU::clear_exclusive() {
    shared_monitor.ClearProcessor(core_id);
}
