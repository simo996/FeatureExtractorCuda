# APPUNTI CUDA

## Intro

Programmazione Cuda è essenzialmente far eseguire alcune funzioni C++, dette kernel, sui numerosi core della GPU secondo l'architettura SIMD (stessa istruzioni su dati multipli). Le funzioni sono self contained: eseguite dai core della GPU senza intervento esterno.

## Architettura Cuda

Logicamente: Grid -> Blocks -> Threads
Fisicamente: Pool di SM -> Pool di Warps

Ogni lancio di Kernel è organizzato in una GRID di BLOCCHI
Ogni blocco è composto da Thread eseguiti in Warp di 32 in parallelo
Ogni SM è organizzato in Warp da 32 ciascuno dei quali esegue 1 thread 

### Thread vs Blocchi

I thread hanno il vantaggio, appartenendo allo stesso blocco, di condividere della memoria e quindi possono fare in modo più efficiente:
* Sincronizzazione
* Comunicazione

### Struttura codice

Il codice è una collezione di funzioni C++; le uniche differenze sono:
* Annotazioni per identificare quali kernel vanno svolti dalla GPU
* Annotazioni per usare le varie memorie offerte dalla GPU

I file CUDA hanno estensione .cu e vengono compilati con nvcc


### PROFILING

Per un programma C/C++ basta:
* Compilare il programma con l'opzione -pg
* Eseguire il programma
* Lanciare gprof e vedere i risultati

Per un programma CUDA basta
* Compilare il programma normalmente
* nvcc file oggetto

### CUDA Limiti di lancio

Dipende dalla "compute capability" della specifica GPU su cui il codice è in esecuzione pertanto è necessario chiedere le proprietà.

Fermi GT540m:
MAI superare il numero di 1024 thread a blocco.
MAI superare il numero di 2^16 blocchi


### Indirizzamento Thread

Ogni thread può determinare il proprio indice sapendo
* in quale blocco gira (blockIdX)
* quanti blocchi ci stanno in un blocco (blockDim)
* il suo indirizzo relativo nel blocco (threadIdx)

int i = blockIdx.x\*blockDim.x + threadIdx.x;

### Lancio di Kernel

Ogni kernel viene lanciato in 1 GRIGLIA di BLOCCHI costituiti da THREAD. Si possono configurare il numero di blocchi e di thread (pur rispettando i vincoli della compute capability):
kernelLaunch<<<NumeroBlocchi, NumeroThreads>>>(params);

kernelLaunch<<<NumeroBlocchi, NumeroThreads, DIM.SHARED>>>(params); quando voglio allocare una quantità dinamica di dati nella shared memory.

I lanci dei kernel sono ASINCRONI pertanto l'host può proseguire anche se non è stata completata l'operazione da parte della gpu

Oss. Prima di lanciare un thread è buona norma settare la GPU con cudaSetDevice(0) per evitare bug nei sistemi con più GPU disponibili [ 1 thread CPU gestisce 1a GPU].

## Memoria

### Tipologia

* Global
	Ha scope globale: la vedono sia Host che Device ( e qualunque sua sotto-entità in qualunque momento)
	__device__ int var1; dichiarazione statica
	cudaMalloc() memoria dinamica
	tempo di vita: pari all'applicazione a meno di cudaFree()
	sulle più recenti compute capabilities è anche cashata (performance gratis)
* Shared 
	Ha scope limitato al blocco (non a Warp, ma rimane consistente per tutto il blocco)
	__shared__ int var
	100x volte più veloce della Global non cachata

### Trasferimenti di memoria

Da Host a Device le informazioni viaggiano via PCI-E ad una velocità 100 volte inferiore a quanto ottenibile tra VRAM e Gpus-> Trasferimenti vanno minimizzati il più possibile; meglio far fare certe operazioni, anche se pesanti e la CPU sarebbe più indicata, in loco pur di evitare il trasferimento. Persino le strutture dati intermedie andrebbero create e usate direttamente dal device

Per trasferire i dati da e verso la GPU c'è un' istruzione: * cudaMemcpy(destination, source, dimension, DIRECTION);

* Destination è il puntatore all'area di memoria in cui verranno messi i dati
* Source è il puntatore all'area di memoria che contiene i dati che vanno trasferiti [flaggare come const]
* Dimension è l'effettiva quantità di byte da trasferire
* Direction è una costante che può essere cudaMemcpyHostToDevice oppure cudaMemcpyDeviceToHost per specificare "il flusso"

** cudaMemcpy() ** è operazione SINCRONA: aspetta che tutto il codice cuda evocato precedentemente concluda e l'esecuzione prosegue solo a trasferimento completato

** cudaMemcpuAsync() ** consente alla CPU di procedere appena viene sottomesso il trasferimento tramite meccanismo simil DMA

https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/ Per Ottimizzare sovrapponendo operazioni

Unified Memory: boh, non è affidabile

### Shared Memory

Quanta memoria del SM viene ripartita tra Shared e cache L1 è programmabile.

BANCHI DI MEMORIA da 32: ottimizzazione ulteriore

### Race Conditions

## Sincronizzazione

### Sincronizzazione Host-Device

I trasferimenti in memoria (con cudamemcpy() ) sono sincroni mentre i lanci di kernel sono asincroni; per forzare la sincronizzazione devo evocare cudaDeviceSynchronize() 

** COSTA ** La sincronizzazione distrugge le performance parallele.

## Sincronizzazione tra Threads

Tipicamente 1 solo thread copia/MODIFICA nella shared memory del blocco dati dalla global memory [letture sono Safe SOLO se non a cavallo di modifiche].

Essendo eseguiti a Warp, i thread raramente vengono eseguiti TUTTI nello stesso istante

Evocando syncthreads() si attende che ciascun thread del blocco abbia completato il proprio lavoro e prevenire data hazards. 
E' una barriera: tutti i thread devono giungere alla riga syncthreads() per procedere. Usare:
* Dopo letture nella shared
* Prima di letture dalla shared

https://devblogs.nvidia.com/using-shared-memory-cuda-cc/

DEADLOCK: si può verificare quando c'è codice Divergente e quindi non tutti i thread evocano la syncthreads()

### Timing delle operazioni

E' possibile creare dei CPU timer prima e dopo l'esecuzione (per assicurarsi che sia completata devo evocare la devicesynchronize () ) MA è una misura affidabile perchè non si ottiene il tempo di effettiva esecuzione ma la differenza tra il tempo di lancio e tempo di fine 

### Device Properties

Buona idea accedere alle device properties per ottenere informazioni sulle capacità tecniche di computazione della GPU attualmente in uso (PER EVITARE CRASH orrendi) tra cui
* numero di blocchi supportati
* numero di thread per blocco supportati 

CODICE
cudaDeviceProp prop;  // STRUCT che conterrà tutte le info
cudaGetDeviceProperties(&prop, i); // leggi e metti in prop le info sulla i-esima GPU del sistema
https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html // campi della struct

## Errori

### Runtime API
TUTTE le chiamate al framework Cuda (API) restituiscono errori da controllare

cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess)
   printf("%s\n", cudaGetErrorString(err));

### Kernels
Kernel vengono eseguiti in modo asincrono e quindi è necessario guardare in una variabile che viene sovrascritta ogni volta che si verifica un errore; tramite  cudaPeekAtLastError() la posso osservare.

### Sincronia

* Sincronia: errori principalmente relativi alle API; kernel non sono neanche partiti
* Asincronia: riguardano crash dei kernel, il controllo è già ritornato alla Cpu pertanto è necessario PRETENDERE la sincronizzazione con cudaDeviceSynchronize() per poterli catturare