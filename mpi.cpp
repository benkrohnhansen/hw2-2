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
    //std::cout << "IN MOVE\n";
    // Boundary conditions
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
	//std::cout << "IN WHILE LOOP\n";
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

double lower_bound;
double upper_bound;
int local_num_parts; // Define a local variable to store the number of particles per rank

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
    MPI_Barrier(MPI_COMM_WORLD);
}
int counter = 0;
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
if (rank == 1) std::cout << "** IN SIMULATE_ONE_STEP ** COUNTER " << counter << std::endl;

    // ============================== MOVE PARTICLES ================================= //
    std::vector<double> ghost_to_above;
    std::vector<double> ghost_to_below;
    std::vector<double> ghost_from_above;
    std::vector<double> ghost_from_below;

    // Define rank above and below
    int rank_above = (rank + 1 < num_procs) ? rank + 1 : -1;
    int rank_below = (rank - 1 >= 0) ? rank - 1 : -1;

    // Iterate over local particles to determine ghost particle counts
    for (size_t i = 0; i < local_parts.size(); i++) {
        if (upper_bound - local_parts[i].y < cutoff) {
            ghost_to_above.push_back(local_parts[i].x);
            ghost_to_above.push_back(local_parts[i].y);
        }
        if (local_parts[i].y - lower_bound < cutoff) {
            ghost_to_below.push_back(local_parts[i].x);
            ghost_to_below.push_back(local_parts[i].y);
        }
    }

    // Set ghost counts based on vector sizes
    int ghost_to_above_count = ghost_to_above.size();
    int ghost_to_below_count = ghost_to_below.size();
    int ghost_from_above_count = 0;
    int ghost_from_below_count = 0;

    // ============================== SEND / RECEIVE GHOST PARTICLE COUNTS ================================= //
    MPI_Request requests[4]; // Array for non-blocking communication

    // Non-blocking send and receive for ghost counts
    MPI_Isend(&ghost_to_above_count, 1, MPI_INT, rank_above, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&ghost_from_below_count, 1, MPI_INT, rank_below, 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(&ghost_to_below_count, 1, MPI_INT, rank_below, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&ghost_from_above_count, 1, MPI_INT, rank_above, 1, MPI_COMM_WORLD, &requests[3]);

    // Wait for all sends/receives to complete
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // Resize vectors based on received counts
    ghost_from_above.resize(ghost_from_above_count);
    ghost_from_below.resize(ghost_from_below_count);

    // ============================== SEND / RECEIVE GHOST PARTICLE DATA ================================= //
    if (ghost_to_above_count > 0 && rank_above >= 0) {
        MPI_Isend(ghost_to_above.data(), ghost_to_above_count, MPI_DOUBLE, rank_above, 2, MPI_COMM_WORLD, &requests[0]);
    }
    if (ghost_from_below_count > 0 && rank_below >= 0) {
        MPI_Irecv(ghost_from_below.data(), ghost_from_below_count, MPI_DOUBLE, rank_below, 2, MPI_COMM_WORLD, &requests[1]);
    }

    if (ghost_to_below_count > 0 && rank_below >= 0) {
        MPI_Isend(ghost_to_below.data(), ghost_to_below_count, MPI_DOUBLE, rank_below, 3, MPI_COMM_WORLD, &requests[2]);
    }
    if (ghost_from_above_count > 0 && rank_above >= 0) {
        MPI_Irecv(ghost_from_above.data(), ghost_from_above_count, MPI_DOUBLE, rank_above, 3, MPI_COMM_WORLD, &requests[3]);
    }

    // Wait for all sends/receives to complete
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    // ============================== PRINT DEBUGGING INFORMATION ================================= //
if (rank == 1 && local_parts.size() > 0) {
    std::cout << "\n[DEBUG] Rank " << rank << " | Particle 0 Info:\n";
    std::cout << "  Position: (" << local_parts[0].x << ", " << local_parts[0].y << ")\n";
    std::cout << "  Velocity: (" << local_parts[0].vx << ", " << local_parts[0].vy << ")\n";
    std::cout << "  Acceleration: (" << local_parts[0].ax << ", " << local_parts[0].ay << ")\n";
}
    // Print all received ghost particle coordinates from above
if (ghost_from_above.size() > 0 && rank == 1) {
    std::cout << "\nReceived " << ghost_from_above.size() / 2
              << " Ghost Particles from Above:\n";
    for (size_t i = 0; i < ghost_from_above.size(); i += 2) {
        std::cout << "  Ghost Particle: x = " << ghost_from_above[i]
                  << ", y = " << ghost_from_above[i + 1] << "\n";
    }
}

// Print all received ghost particle coordinates from below
if (ghost_from_below.size() > 0 && rank == 1) {
    std::cout << "\nReceived " << ghost_from_below.size() / 2
              << " Ghost Particles from Below:\n";
    for (size_t i = 0; i < ghost_from_below.size(); i += 2) {
        std::cout << "  Ghost Particle: x = " << ghost_from_below[i]
                  << ", y = " << ghost_from_below[i + 1] << "\n";
    }

}

        // ============================= Compute Forces ============================= //
    for (int i = 0; i < local_parts.size(); ++i) {
        local_parts[i].ax = local_parts[i].ay = 0;
        for (int j = 0; j < local_parts.size(); ++j) {
            apply_force(local_parts[i], local_parts[j]);
        }
    // Compute forces with ghost particles from above
    for (size_t jj = 0; jj < ghost_from_above.size(); jj += 2) {
        apply_force(local_parts[i], ghost_from_above[jj], ghost_from_above[jj + 1]);
    }

    // Compute forces with ghost particles from below
    for (size_t jj = 0; jj < ghost_from_below.size(); jj += 2) {
        apply_force(local_parts[i], ghost_from_below[jj], ghost_from_below[jj + 1]);
    }
}

// ============================== MOVE PARTICLES ============================== //
for (size_t i = 0; i < local_parts.size(); i++) {
    move(local_parts[i], size);
}
MPI_Barrier(MPI_COMM_WORLD);
counter++;
}
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    // Print the x coordinate of the first particle
}
