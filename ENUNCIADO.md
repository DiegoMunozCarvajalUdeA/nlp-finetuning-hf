# Proyecto 2 del semestre: Fine-tuning para tareas de NLP

Instrucciones:

1. Crear una cuenta de usuario en Hugging Face. https://huggingface.co/join

2. Seleccionar tres bases de datos, una por cada uno de los siguientes tipo de problema de NLP:

   - Sequence classification (sentiment analysis, spam detection, etc.)
   - Sequence-labeling (Part-of-Speech-Tagging, Named Entity Recognition, etc.)
   - Sequence-to-Sequence (Machine Translation, Summarization, etc.)

   Las bases de datos las pueden tomar de Hugging Face para facilidad en el desarrollo del proyecto. https://huggingface.co/datasets

3. Crear un archivo .ipynb (Jupyter notebook en Anaconda o Google Colab) para el desarrollo del proyecto.

4. Dentro del notebook describir en detalle la base de datos de cada problema que va a utilizar en su proyecto. Debe quedar claro el dominio del problema, la utilidad, los datos de entrada, la variable de salida, el tipo de problema, etc.

5. Realizar un análisis exploratorio de los datos de su base de datos en el que se observen un par de ejemplos que permitan entender el problema abordado.

6. Aplicar la técnica de fine-tuning sobre un modelo tipo BERT, GPT o BART (T5), según sea el caso, para entrenar su propio modelo y resolver el problema seleccionado. Ayuda: https://huggingface.co/learn/llm-course/chapter1/5

7. Probar diferentes valores de hiperparámetros (el learning rate es clave) con un sample manejable y quedarse con el mejor de los casos para entrenar el modelo con la data completa (Validar hasta donde les da la GPU de Colab y limitar el dataset si es estrictamente necesario). Tenga en cuenta que los modelos pueden tardar horas en el entrenamiento.

8. Mostrar el mejor modelo encontrado para solucionar cada uno de los problemas abordados.

9. Suba el notebook a su cuenta de github con un Readme sencillo describiendo el trabajo realizado.

10. Publicar un post en su cuenta de LinkedIn mostrando el desarrollo del proyecto y dejando un enlace al repositorio del proyecto en Github.

11. Hágase publicidad en su hoja de vida, mostrando que sabe de NLP teórico y práctico, siendo sincero de acuerdo a lo que ha aprendido en el curso, no infle su CV, pero muestre a los demás lo que sabe.

Nota: el entregable del proyecto debe ser un enlace al repositorio de github donde se encuentre la descripción del proyecto y el archivo notebook con todo el desarrollo.

Sustentación obligatoria: enviar un video, con cámara encendida donde se vea usted presentando, de máximo 12 minutos (4 minutos por tarea) antes de la fecha de cierre de la tarea. Presencial.
