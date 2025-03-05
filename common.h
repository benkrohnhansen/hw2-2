#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <cstdint>
#include <mpi.h>

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

// Particle Data Structure
typedef struct particle_t {
    uint64_t id; // Particle ID
    double x;    // Position X
    double y;    // Position Y
    double vx;   // Velocity X
    double vy;   // Velocity Y
    double ax;   // Acceleration X
    double ay;   // Acceleration Y
} particle_t;

extern MPI_Datatype PARTICLE;

// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs);
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs,
    double& ghost_calc, double& time_ghost_cnt_comm, double& ghost_cnt_waitall, 
    double& ghost_comm, double& ghost_waitall, double& force_calc, double& move, 
    double& calc_particle_mv, double& part_cnt_waitall, double& part_cnt, 
    double& part_waitall, double& part, double& part_insert, double& end_barrier);
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs, 
    double& sort_time, double& gather_time);

#endif
