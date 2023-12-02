#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

__host__ void fu(float a, float b, float c) 
{
    std::cout << "x: " << std::endl;  
}

__global__ void particle_kernel(particles* d_part, EMfield* d_field, grid* d_grd, parameters* d_param) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_part->nop) return;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) d_param->dt/((double) d_part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = d_part->qom*dto2/d_param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = d_part->x[idx];
    yptilde = d_part->y[idx];
    zptilde = d_part->z[idx];
    // calculate the average velocity iteratively
    for(int innter=0; innter < d_part->NiterMover; innter++){
        // interpolation G-->P
        ix = 2 +  int((d_part->x[idx] - d_grd->xStart)*d_grd->invdx);
        iy = 2 +  int((d_part->y[idx] - d_grd->yStart)*d_grd->invdy);
        iz = 2 +  int((d_part->z[idx] - d_grd->zStart)*d_grd->invdz);
        
        // calculate weights
        xi[0]   = d_part->x[idx] - d_grd->XN_flat[((ix - 1) * d_grd->nyn * d_grd-> nzn) + (iy * d_grd->nzn) + iz];
        eta[0]  = d_part->y[idx] - d_grd->YN_flat[(ix * d_grd->nyn * d_grd-> nzn) + ((iy - 1) * d_grd->nzn) + iz];
        zeta[0] = d_part->z[idx] - d_grd->ZN_flat[(ix * d_grd->nyn * d_grd-> nzn) + (iy * d_grd->nzn) + iz - 1];

        xi[1]   = d_grd->XN_flat[(ix * d_grd->nyn * d_grd-> nzn) + (iy * d_grd->nzn) + iz] - d_part->x[idx];
        eta[1]  = d_grd->YN_flat[(ix * d_grd->nyn * d_grd-> nzn) + (iy * d_grd->nzn) + iz] - d_part->y[idx];
        zeta[1] = d_grd->ZN_flat[(ix * d_grd->nyn * d_grd-> nzn) + (iy * d_grd->nzn) + iz] - d_part->z[idx];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * d_grd->invVOL;
        
        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
        
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    Exl += weight[ii][jj][kk]*d_field->Ex_flat[((ix-ii)* d_grd->nyn * d_grd->nzn) + ((iy-jj) * d_grd->nzn) + (iz-kk)];
                    Eyl += weight[ii][jj][kk]*d_field->Ey_flat[((ix-ii)* d_grd->nyn * d_grd->nzn) + ((iy-jj) * d_grd->nzn) + (iz-kk)];
                    Ezl += weight[ii][jj][kk]*d_field->Ez_flat[((ix-ii)* d_grd->nyn * d_grd->nzn) + ((iy-jj) * d_grd->nzn) + (iz-kk)];
                    Bxl += weight[ii][jj][kk]*d_field->Bxn_flat[((ix-ii)* d_grd->nyn * d_grd->nzn) + ((iy-jj) * d_grd->nzn) + (iz-kk)];
                    Byl += weight[ii][jj][kk]*d_field->Byn_flat[((ix-ii)* d_grd->nyn * d_grd->nzn) + ((iy-jj) * d_grd->nzn) + (iz-kk)];
                    Bzl += weight[ii][jj][kk]*d_field->Bzn_flat[((ix-ii)* d_grd->nyn * d_grd->nzn) + ((iy-jj) * d_grd->nzn) + (iz-kk)];
                }
        
        // end interpolation
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= d_part->u[idx] + qomdt2*Exl;
        vt= d_part->v[idx] + qomdt2*Eyl;
        wt= d_part->w[idx] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
        // update position
        d_part->x[idx] = xptilde + uptilde*dto2;
        d_part->y[idx] = yptilde + vptilde*dto2;
        d_part->z[idx] = zptilde + wptilde*dto2;            
        
    } // end of iteration

    // update the final position and velocity
    d_part->u[idx]= 2.0*uptilde - d_part->u[idx];
    d_part->v[idx]= 2.0*vptilde - d_part->v[idx];
    d_part->w[idx]= 2.0*wptilde - d_part->w[idx];
    d_part->x[idx] = xptilde + uptilde*dt_sub_cycling;
    d_part->y[idx] = yptilde + vptilde*dt_sub_cycling;
    d_part->z[idx] = zptilde + wptilde*dt_sub_cycling;
                                
    // X-DIRECTION: BC particles
    if (d_part->x[idx] > d_grd->Lx){
        if (d_param->PERIODICX==true){ // PERIODIC
            d_part->x[idx] = d_part->x[idx] - d_grd->Lx;
        } else { // REFLECTING BC
            d_part->u[idx] = -d_part->u[idx];
            d_part->x[idx] = 2*d_grd->Lx - d_part->x[idx];
        }
    }
                                                                
    if (d_part->x[idx] < 0){
        if (d_param->PERIODICX==true){ // PERIODIC
            d_part->x[idx] = d_part->x[idx] + d_grd->Lx;
        } else { // REFLECTING BC
            d_part->u[idx] = -d_part->u[idx];
            d_part->x[idx] = -d_part->x[idx];
        }
    }
    
    // Y-DIRECTION: BC particles
    if (d_part->y[idx] > d_grd->Ly){
        if (d_param->PERIODICY==true){ // PERIODIC
            d_part->y[idx] = d_part->y[idx] - d_grd->Ly;
        } else { // REFLECTING BC
            d_part->v[idx] = -d_part->v[idx];
            d_part->y[idx] = 2*d_grd->Ly - d_part->y[idx];
        }
    }
                                                                
    if (d_part->y[idx] < 0){
        if (d_param->PERIODICY==true){ // PERIODIC
            d_part->y[idx] = d_part->y[idx] + d_grd->Ly;
        } else { // REFLECTING BC
            d_part->v[idx] = -d_part->v[idx];
            d_part->y[idx] = -d_part->y[idx];
        }
    }
                                                                
    // Z-DIRECTION: BC particles
    if (d_part->z[idx] > d_grd->Lz){
        if (d_param->PERIODICZ==true){ // PERIODIC
            d_part->z[idx] = d_part->z[idx] - d_grd->Lz;
        } else { // REFLECTING BC
            d_part->w[idx] = -d_part->w[idx];
            d_part->z[idx] = 2*d_grd->Lz - d_part->z[idx];
        }
    }
                                                                
    if (d_part->z[idx] < 0){
        if (d_param->PERIODICZ==true){ // PERIODIC
            d_part->z[idx] = d_part->z[idx] + d_grd->Lz;
        } else { // REFLECTING BC
            d_part->w[idx] = -d_part->w[idx];
            d_part->z[idx] = -d_part->z[idx];
        }
    }
}


// Copies result back to CPU and deallocates
void retrieve_particles(struct particles* d_part, struct particles* h_part) {

  // Get result
    // Copy
  cudaMemcpy(h_part->x, d_part->x, h_part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_part->y, d_part->y, h_part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_part->z, d_part->z, h_part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_part->u, d_part->u, h_part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_part->v, d_part->v, h_part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_part->w, d_part->w, h_part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

  // Deallocate
  cudaFree(d_part->x);
  cudaFree(d_part->y);
  cudaFree(d_part->z);
  cudaFree(d_part->u);
  cudaFree(d_part->v);
  cudaFree(d_part->w);
}

void h_part(particles* h_part, particles* d_prp_part, particles** d_part_ptr) {
  memcpy(d_prp_part, h_part, sizeof(particles));
  
  cudaMalloc(&d_prp_part->x, h_part->npmax * sizeof(FPpart));
  cudaMalloc(&d_prp_part->y, h_part->npmax * sizeof(FPpart));
  cudaMalloc(&d_prp_part->z, h_part->npmax * sizeof(FPpart));
  cudaMalloc(&d_prp_part->u, h_part->npmax * sizeof(FPpart));
  cudaMalloc(&d_prp_part->v, h_part->npmax * sizeof(FPpart));
  cudaMalloc(&d_prp_part->w, h_part->npmax * sizeof(FPpart));
  cudaMalloc(d_part_ptr, sizeof(particles));

  cudaMemcpy(d_prp_part->x, h_part->x, h_part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prp_part->y, h_part->y, h_part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prp_part->z, h_part->z, h_part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prp_part->u, h_part->u, h_part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prp_part->v, h_part->v, h_part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prp_part->w, h_part->w, h_part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
  cudaMemcpy(*d_part_ptr, d_prp_part, sizeof(particles), cudaMemcpyHostToDevice);
}


int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    int TPB = 512;
    int blocks = (part->nop - 1) / TPB + 1;

    EMfield* d_field; 
    grid* d_grid; 
    parameters* d_param;

    cudaMalloc(&d_grid, sizeof(grid));
    cudaMemcpy(d_grid, grd,sizeof(grid),cudaMemcpyHostToDevice);

    cudaMalloc(&d_field, sizeof(EMfield));
    cudaMemcpy(d_field, field,sizeof(EMfield),cudaMemcpyHostToDevice);

    cudaMalloc(&d_param, sizeof(parameters));
    cudaMemcpy(d_param, param,sizeof(parameters),cudaMemcpyHostToDevice);

    // Copy arrays of field struct
    FPfield *d_field_Ex_flat, *d_field_Ey_flat, *d_field_Ez_flat, *d_field_Bxn_flat, *d_field_Byn_flat, *d_field_Bzn_flat;
    cudaMalloc(&d_field_Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_field_Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_field_Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_field_Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_field_Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_field_Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(d_field_Ex_flat, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field_Ey_flat, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field_Ez_flat, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field_Bxn_flat, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field_Byn_flat, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field_Bzn_flat, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    // Binding pointers with d_prp_part struct
    cudaMemcpy(&(d_field->Ex_flat), &d_field_Ex_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_field->Ey_flat), &d_field_Ey_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_field->Ez_flat), &d_field_Ez_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_field->Bxn_flat), &d_field_Bxn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_field->Byn_flat), &d_field_Byn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_field->Bzn_flat), &d_field_Bzn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);

    // Copy arrays of grd struct
    FPfield *d_XN_flat, *d_YN_flat, *d_ZN_flat;
    cudaMalloc(&d_XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&d_ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(d_XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    // Binding pointers with d_prp_part struct
    cudaMemcpy(&(d_grid->XN_flat), &d_XN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_grid->YN_flat), &d_YN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_grid->ZN_flat), &d_ZN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);

    particles d_prp_part;
    particles* d_part_ptr;
    h_part(part, &d_prp_part, &d_part_ptr);

    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
    // std::cout << "creating kernel with: " << blocks_in_grid << ":" << threads_in_block << std::endl;
        particle_kernel<<<blocks, TPB>>>(
            d_part_ptr,
            d_field,
            d_grid,
            d_param
        );

        gpuErrchk(cudaDeviceSynchronize());
    
    }

    retrieve_particles(&d_prp_part, part);

    cudaFree(d_XN_flat);
    cudaFree(d_YN_flat);
    cudaFree(d_ZN_flat);

    cudaFree(d_field_Ex_flat);
    cudaFree(d_field_Ey_flat);
    cudaFree(d_field_Ez_flat);
    cudaFree(d_field_Bxn_flat);
    cudaFree(d_field_Byn_flat);
    cudaFree(d_field_Bzn_flat);
  
    cudaFree(d_param);     
    cudaFree(d_grid);
    cudaFree(d_field);   
    cudaFree(d_part_ptr);                     

    return(0); // exit succcesfully
}


/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields

        std::cout << part->nop << ":" << part->NiterMover << std::endl; 
            
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
