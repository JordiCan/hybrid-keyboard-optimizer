# Hybrid Keyboard Optimizer

Un sistema de optimización híbrido para el diseño de distribuciones de teclado que combina algoritmos genéticos y recocido simulado para minimizar la fatiga en la escritura y maximizar la eficiencia.

## Motivación

El diseño QWERTY data de 1870 y fue creado para evitar atascos mecánicos en las máquinas de escribir, no para optimizar la ergonomía humana. Este proyecto utiliza técnicas de optimización algorítmica para explorar distribuciones alternativas basadas en patrones de escritura modernos y análisis de frecuencia de caracteres.

## Metodología

El sistema implementa dos algoritmos metaheurísticos:

**Algoritmos Genéticos (GA)**
Genera poblaciones de distribuciones de teclado, selecciona las más eficientes mediante una función de fitness, y produce nuevas generaciones a través de operadores de cruce y mutación.

**Recocido Simulado (SA)**
Refina las soluciones candidatas mediante un proceso de enfriamiento gradual que acepta temporalmente configuraciones subóptimas para escapar de mínimos locales.

**Enfoque Híbrido**
La combinación de ambos métodos permite una exploración amplia del espacio de soluciones (GA) seguida de un refinamiento local (SA), resultando en mejores soluciones que las obtenidas por cada algoritmo individualmente.

## Función de Fitness

El sistema evalúa cada distribución según múltiples criterios:

- **Distancia de desplazamiento**: Distancia total recorrida por los dedos
- **Alternancia entre manos**: Frecuencia de cambio entre mano izquierda y derecha
- **Penalización por mismo dedo**: Uso consecutivo del mismo dedo
- **Uso de fila base**: Porcentaje de pulsaciones en la fila principal
- **Optimización de bigramas**: Colocación eficiente de pares de letras frecuentes
- **Balance entre manos**: Distribución equitativa de la carga de trabajo

## Instalación

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Dependencias: numpy, matplotlib, jupyter

## Uso

Ejecutar los notebooks de Jupyter:

```bash
jupyter notebook
```

Archivos disponibles:
- `cross_genetic_algorithm.ipynb` - Implementación del algoritmo genético
- `cross_simulated_annealing.ipynb` - Implementación del recocido simulado

Los notebooks incluyen implementaciones completas, visualizaciones del proceso de optimización y comparaciones con distribuciones establecidas (QWERTY, Dvorak, Colemak).

## Visualizaciones

El proyecto genera:
- Mapas de calor de uso por dedo
- Gráficas de convergencia del algoritmo
- Comparativas de rendimiento
- Métricas de distancia de desplazamiento

## Consideraciones

- La distribución óptima depende del idioma, tipo de contenido y preferencias individuales
- Pequeñas mejoras en eficiencia pueden requerir cambios significativos en la distribución
- La memoria muscular representa un obstáculo práctico para la adopción de nuevas distribuciones
- Diferentes contextos de uso (programación vs. escritura en prosa) pueden beneficiarse de distribuciones distintas

## Aplicación Académica

Este proyecto resulta útil para:
- Estudio de algoritmos de optimización metaheurística
- Investigación en interacción humano-computadora
- Análisis de diseño ergonómico
- Aplicación práctica de técnicas de inteligencia computacional

## Contribuciones

Las contribuciones son bienvenidas. Áreas de interés:
- Mejoras en la función de fitness
- Nuevas estrategias de optimización
- Soporte para múltiples idiomas
- Validación experimental con usuarios reales

