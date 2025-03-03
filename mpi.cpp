#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
// Put any static global variables here that you will use throughout the simulation.

// Apply force between two particles
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);  // Ensure minimum distance between particles
    double r = sqrt(r2);  // Compute distance between particles

    // Short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void apply_force(particle_t& particle, double& neighbor_x, double& neighbor_y) {
    // Calculate Distance
    double dx = neighbor_x - particle.x;
    double dy = neighbor_y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);  // Ensure minimum distance between particles
    double r = sqrt(r2);  // Compute distance between particles

    // Short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Boundary conditions
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

double lower_bound;
double upper_bound;
int local_num_parts; // Define a local variable to store the number of particles per rank

int printingRank = 0; // Set this to control which rank prints debug output

std::vector<particle_t> local_parts;

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_height = size / num_procs;
    lower_bound = rank * row_height;
    upper_bound = (rank + 1) * row_height;
    
    // Store local particles relevant to this rank
    local_parts.clear();
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].y >= lower_bound && parts[i].y < upper_bound) {
            local_parts.push_back(parts[i]);
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    
    // ============================== MOVE PARTICLES ================================= //
    for (size_t i = 0; i < local_parts.size(); i++) {
        move(local_parts[i], local_parts.size());
    }

    // ============================== SEND / RECEIVE GHOST PARTICLE COUNTS ================================= //
    int ghost_to_above_count = 0;
    int ghost_to_below_count = 0;
    int ghost_from_above_count = 0;
    int ghost_from_below_count = 0;

    // Define rank above and below
    int rank_above = (rank + 1 < num_procs) ? rank + 1 : -1;
    int rank_below = (rank - 1 >= 0) ? rank - 1 : -1;

    // Iterate over local particles to determine ghost particle counts
    for (size_t i = 0; i < local_parts.size(); i++) {
        if (upper_bound - local_parts[i].y < cutoff) {
            ghost_to_above_count++;
        }
        if (local_parts[i].y - lower_bound < cutoff) {
            ghost_to_below_count++;
        }
    }

    // First: Receive from rank below, send to rank below
    if (rank_below != -1) {
        MPI_Sendrecv(&ghost_to_below_count, 1, MPI_INT, rank_below, 0,
                     &ghost_from_below_count, 1, MPI_INT, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "[Rank " << rank << "] Sent " << ghost_to_below_count
                  << " ghost particles to Rank " << rank_below 
                  << " | Received " << ghost_from_below_count 
                  << " ghost particles from Rank " << rank_below << std::endl;
    }

    // Second: Send to rank above, receive from rank above
    if (rank_above != -1) {
        MPI_Sendrecv(&ghost_to_above_count, 1, MPI_INT, rank_above, 1,
                     &ghost_from_above_count, 1, MPI_INT, rank_above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "[Rank " << rank << "] Sent " << ghost_to_above_count
                  << " ghost particles to Rank " << rank_above 
                  << " | Received " << ghost_from_above_count 
                  << " ghost particles from Rank " << rank_above << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    // Print the x coordinate of the first particle
}