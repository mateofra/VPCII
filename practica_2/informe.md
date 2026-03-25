# Memoria final del Proyecto 2

## Vision por Computador II

- Asignatura: Vision por Computador II
- Curso: 2025-2026
- Universidad: Universidade de Santiago de Compostela
- Titulo del proyecto: Editor de imagenes inteligente
- Repositorio de trabajo: VPCII
- Notebook principal: practica_2/VPCII_2.ipynb

## 1. Resumen

En este proyecto se desarrolla una base funcional de un editor de imagenes inteligente sobre COCO, centrado en dos fases principales del enunciado:

1. Fase I: generacion de imagenes desde ruido con un enfoque DDPM ligero.
2. Fase II: generacion condicionada por texto usando Stable Diffusion y analisis del efecto de CFG.

Ademas, se ha preparado una estructura de evaluacion reproducible con artefactos de salida, metricas (FID y CLIP Score) y checklist de cobertura para la entrega.

## 2. Objetivos

Los objetivos de trabajo abordados en esta memoria son:

1. Construir un flujo reproducible de preparacion del dominio desde COCO.
2. Estandarizar la evaluacion visual de la Fase I con grids y proceso de denoising.
3. Incorporar evaluacion cuantitativa en Fase I con FID.
4. Implementar en Fase II un barrido de CFG con prompt y semilla fijos.
5. Medir similitud texto-imagen con CLIP Score.
6. Dejar listo el material para defensa y entrega final.

## 3. Dataset y preprocesado

Se utiliza el dataset COCO (val2017 + anotaciones trainval2017) con este flujo:

1. Descarga de imagenes y anotaciones.
2. Carga de instances_val2017.json.
3. Seleccion de una supercategoria (por defecto: animal).
4. Filtrado de imagenes que contienen categorias de dicha supercategoria.
5. Copia del subconjunto en dataset_fase1 para entrenamiento y comparativas.

Este preprocesado permite trabajar con un dominio concreto y facilita experimentacion iterativa.

## 4. Metodologia

### 4.1 Fase I: motor base de generacion

Se habilitan bloques para:

1. Generar grid de 16 imagenes reales.
2. Generar grid de 16 imagenes sinteticas.
3. Visualizar denoising iterativo a partir de pasos guardados (step_*.png).
4. Calcular FID entre conjunto real y generado.

Ademais, para facilitar a execucion en Colab, engadense dous modos de traballo:

1. `smoke`: validacion rapida do pipeline (menos pasos e menos imaxes).
2. `full`: adestramento orientado a resultados finais de entrega.

No estado actual da implementacion, a Fase I inclue unha **ablation completa** e un
**ultimo run de avaliacion con todas as melloras activadas**.

Configuracions experimentais executadas:

1. `baseline_linear`
2. `aug_only`
3. `aug_attn_cosine`
4. `all_mejoras_final` (run final)

Melloras incorporadas no run final:

1. Data Augmentation
2. Self-Attention no bottleneck da U-Net
3. Scheduler cosine
4. EMA de pesos
5. Mixed Precision (AMP)
6. Gradient accumulation
7. Noise offset no ruído de adestramento
8. Maior capacidade do modelo (`base` superior) en modo `full`
9. Maior número de timesteps en modo `full`

Salidas esperadas en practica_2/resultados:

- fase1_grid_16_reales.png
- fase1_grid_16_sinteticas.png
- fase1_denoising_steps.png
- metricas/fid.json

### 4.2 Fase II: condicionamiento por texto

Se habilitan bloques para:

1. Extraer prompts de captions_val2017 asociados al dominio seleccionado.
2. Generar imagenes con Stable Diffusion variando CFG y manteniendo prompt + seed.
3. Construir tabla visual comparativa del barrido CFG.
4. Calcular CLIP Score para las imagenes generadas frente al prompt base.

Salidas esperadas en practica_2/resultados:

- fase2_cfg_sweep.png
- fase2_cfg/cfg_*.png
- metricas/clip_score.json

## 5. Implementacion tecnica

La implementacion en el notebook incluye:

1. Inicializacion de rutas y carpetas de salida.
2. Funciones reutilizables para listado de imagenes y trazado de grids.
3. Funcion de FID con InceptionV3 + distancia de Frechet.
4. Funcion de extraccion de prompts desde anotaciones de COCO.
5. Funcion de generacion con Stable Diffusion para barrido CFG.
6. Funcion de CLIP Score con transformers.
7. Estructura CHECKLIST para control de avance de entregables.
8. Perfilado de adestramento en Fase I mediante `RUN_MODE` (`smoke`/`full`).
9. Reutilizacion de funcions clave de `proyecto_2.ipynb`.

Funcions de `proyecto_2.ipynb` reutilizadas explicitamente en `VPCII_2.ipynb`:

- `linear_beta_schedule`
- `forward_diffusion`
- `sample_timesteps`
- `preprocess_for_inception`
- `get_inception_features`
- `calculate_fid`

Dependencias principales:

- numpy
- matplotlib
- pillow
- torch
- torchvision
- scipy
- diffusers
- transformers
- accelerate

## 6. Resultados y evidencias

En el estado actual, la memoria deja definida la canalizacion completa y los artefactos de salida para la evaluacion. Los valores numericos finales dependen de ejecutar entrenamiento/generacion en el entorno con recursos adecuados.

Estado de implementacion actualizado:

1. Fase I completada con DDPM lixeiro e experimento base/mellora.
2. FID migrado a pipeline Inception (coherente co enfoque de `proyecto_2.ipynb`).
3. Modo `smoke/full` operativo para acelerar probas en Colab e escalar a entrega final.
4. Ablation multi-configuracion implementada para medir impacto de cada mellora.
5. Run final `all_mejoras_final` aplicado como referencia para grid, denoising e FID.

Plantilla de resultados para completar tras ejecucion:

- FID (Fase I): pendiente de ejecucion
- CLIP Score medio (Fase II): pendiente de ejecucion
- CLIP Score desviacion (Fase II): pendiente de ejecucion

Evidencias visuales objetivo:

1. Comparativa 16 reales vs 16 sinteticas.
2. Evolucion de denoising por pasos.
3. Tabla visual de CFG con semilla fija.

## 7. Discusion

Fortalezas del enfoque implementado:

1. Flujo modular y reproducible.
2. Separacion clara entre evaluacion visual y cuantitativa.
3. Facil extension a variantes (scheduler, atencion, augmentation, prompts).

Riesgos y limitaciones:

1. El coste computacional puede ser alto para FID/SD en CPU.
2. La calidad final depende de entrenamiento efectivo del modelo generativo.
3. El CLIP Score no sustituye evaluacion humana cualitativa.

## 8. Conclusiones

Se ha dejado preparada una memoria de trabajo alineada con el enunciado, junto con una estructura practica para:

1. Cubrir los requisitos nucleares de Fase I y Fase II.
2. Generar automaticamente figuras de comparativa y metricas.
3. Facilitar la defensa con resultados trazables y reproducibles.

Como siguiente paso inmediato, basta ejecutar de forma ordenada las celdas del notebook, consolidar metricas finales y trasladar resultados a la presentacion.

## 9. Trabajo futuro

1. Incorporar variantes de DDPM (attention y distintos schedulers) y comparar FID.
2. Añadir evaluacion por lotes de prompts y analisis estadistico de CFG.
3. Integrar Fase III cuando se concrete en clase (control por anotaciones y edicion localizada).
4. Automatizar generacion de resumen para diapositivas desde los JSON de metricas.

## 10. Checklist de entrega

- Codigo fuente en notebook: completo
- Secciones Fase I/Fase II: completas
- Plantillas de metricas y figuras: completas
- Valores finales de metricas: pendiente de ejecucion
- Diapositivas PDF de defensa: pendiente

## Anexo A. Guia minima de ejecucion

Orden recomendado:

1. Descargar y descomprimir COCO.
2. Filtrar dominio y crear dataset_fase1.
3. Generar imagenes sinteticas de Fase I y pasos de denoising.
4. Ejecutar grid + FID.
5. Extraer prompts y ejecutar barrido CFG.
6. Ejecutar CLIP Score.
7. Revisar CHECKLIST y consolidar resultados para defensa.
