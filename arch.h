#include "config.h"	// From configured OpenBLAS, here we take CPU type

#if defined(OS_SUNOS)
#define YIELDING	thr_yield()
#endif

#if defined(OS_WINDOWS)
#if defined(_MSC_VER) && !defined(__clang__)
#define YIELDING    YieldProcessor()
#else
#define YIELDING	SwitchToThread()
#endif
#endif

#if defined(ARMV7) || defined(ARMV6) || defined(ARMV8) || defined(ARMV5)
#define YIELDING        asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop; \n");
#endif

#ifdef BULLDOZER
#ifndef YIELDING
#define YIELDING        __asm__ __volatile__ ("nop;nop;nop;nop;nop;nop;nop;nop;\n");
#endif
#endif


#ifdef PILEDRIVER
#ifndef YIELDING
#define YIELDING        __asm__ __volatile__ ("nop;nop;nop;nop;nop;nop;nop;nop;\n");
#endif
#endif

/*
#ifdef STEAMROLLER
#ifndef YIELDING
#define YIELDING        __asm__ __volatile__ ("nop;nop;nop;nop;nop;nop;nop;nop;\n");
#endif
#endif
*/

#ifndef YIELDING
#define YIELDING	sched_yield()
#endif

#if defined ARCH_ARM || defined ARCH_ARM64
#define WMB  __asm__ __volatile__ ("dmb  ishst" : : : "memory")
#elif defined sparc
#define WMB __asm__ __volatile__ ("nop")
#elif defined ARCH_ALPHA
#define WMB asm("wmb")
#elif defined ARCH_POWER
#define WMB __asm__ __volatile__ ("sync")
#else
#define WMB
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define CONSTRUCTOR __cdecl
#define DESTRUCTOR __cdecl
#else
#define CONSTRUCTOR	__attribute__ ((constructor))
#define DESTRUCTOR	__attribute__ ((destructor))
#endif

#if defined(_MSC_VER) && !defined(__clang__) && !defined DODOUBLE
#include <windows.h>
BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
  switch (ul_reason_for_call)
  {
    case DLL_PROCESS_ATTACH:
      gemmconv_init();
      break;
    case DLL_THREAD_ATTACH:
      break;
    case DLL_THREAD_DETACH:
      break;
    case DLL_PROCESS_DETACH:
      gemmconv_exit();
      break;
    default:
      break;
  }
  return TRUE;
}
#endif
