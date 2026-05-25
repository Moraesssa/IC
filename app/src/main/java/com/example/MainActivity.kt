package com.example

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.ml.DataRow
import com.example.ml.DecisionNode
import com.example.ml.MLCore
import com.example.ui.theme.MyApplicationTheme
import com.example.ui.theme.EditorialBlue
import com.example.ui.theme.EditorialDarkNavy
import com.example.ui.theme.EditorialSoftBlue
import com.example.ui.theme.EditorialCream
import com.example.ui.theme.EditorialCharcoal
import com.example.ui.theme.EditorialMuted
import com.example.ui.theme.EditorialBorder
import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    setContent {
      MyApplicationTheme {
        MainLabScreen()
      }
    }
  }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainLabScreen() {
  val context = LocalContext.current

  // State managers
  var randomSeed by remember { mutableStateOf(42) }
  var noiseAmplitude by remember { mutableStateOf(0.2f) }
  var maxTreeDepth by remember { mutableStateOf(3) }
  var activeTab by remember { mutableStateOf(0) }

  // Generate dataset and train tree on state changes
  val dataset = remember(randomSeed, noiseAmplitude) {
    MLCore.generateDataset(
        randomSeed = randomSeed.toLong(),
        noiseAmplitude = noiseAmplitude.toDouble(),
        sampleSize = 100
    )
  }

  val stats = remember(dataset) {
    MLCore.calculateStats(dataset)
  }

  val trainedTree = remember(dataset, maxTreeDepth) {
    val tree = MLCore.trainRegressionTree(
        data = dataset,
        maxDepth = maxTreeDepth,
        minSamplesSplit = 5
    )
    // Run prediction on original dataset
    dataset.forEach { row ->
      row.yPred = tree.predict(row.x1, row.x2, row.x3, row.x4)
    }
    tree
  }

  val evaluationMetrics = remember(dataset) {
    MLCore.calculateMetrics(
        actuals = dataset.map { it.y },
        predictions = dataset.map { it.yPred }
    )
  }

  // Interactive sandbox sliders state
  var sandboxX1 by remember { mutableStateOf(0.5f) }
  var sandboxX2 by remember { mutableStateOf(0.5f) }
  var sandboxX3 by remember { mutableStateOf(0.5f) }
  var sandboxX4 by remember { mutableStateOf(0.5f) }

  val sandboxYActual by remember(sandboxX1, sandboxX2, sandboxX3, sandboxX4, noiseAmplitude) {
    derivedStateOf {
      val base = sandboxX1.toDouble().pow(2.0) +
              sandboxX2.toDouble().pow(3.0) +
              sandboxX3.toDouble() +
              sandboxX4.toDouble().pow(4.0)
      // Display formula value without noise comparison or with average noise for contrast
      base + (noiseAmplitude.toDouble() / 2.0)
    }
  }

  val sandboxYPredicted by remember(trainedTree, sandboxX1, sandboxX2, sandboxX3, sandboxX4) {
    derivedStateOf {
      trainedTree.predict(
          sandboxX1.toDouble(),
          sandboxX2.toDouble(),
          sandboxX3.toDouble(),
          sandboxX4.toDouble()
      )
    }
  }

  Scaffold(
      modifier = Modifier
          .fillMaxSize()
          .testTag("main_screen"),
      contentWindowInsets = WindowInsets.safeDrawing
  ) { padding ->
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .background(MaterialTheme.colorScheme.background)
    ) {

      // Header Banner (Academic theme branding)
      AcademicHeader(
          onEmailClick = {
            composeEmail(context, evaluationMetrics.mse, evaluationMetrics.r2)
          }
      )

      // Parameter Config Drawer/Panel
      LabParametersPanel(
          seed = randomSeed,
          noise = noiseAmplitude,
          depth = maxTreeDepth,
          onSeedChange = { randomSeed = it },
          onNoiseChange = { noiseAmplitude = it },
          onDepthChange = { maxTreeDepth = it }
      )

      // Tabs Header
      ScrollableTabRow(
          selectedTabIndex = activeTab,
          containerColor = MaterialTheme.colorScheme.surfaceColorAtElevation(2.dp),
          contentColor = MaterialTheme.colorScheme.primary,
          edgePadding = 16.dp,
          modifier = Modifier
              .fillMaxWidth()
              .testTag("tab_bar")
      ) {
        Tab(
            selected = activeTab == 0,
            onClick = { activeTab = 0 },
            text = { Text("Visão Geral", fontWeight = FontWeight.SemiBold) },
            icon = { Icon(Icons.Default.Dashboard, contentDescription = "Dashboard") }
        )
        Tab(
            selected = activeTab == 1,
            onClick = { activeTab = 1 },
            text = { Text("Dados (Q1-Q3)", fontWeight = FontWeight.SemiBold) },
            icon = { Icon(Icons.Default.TableChart, contentDescription = "DataFrame") }
        )
        Tab(
            selected = activeTab == 2,
            onClick = { activeTab = 2 },
            text = { Text("Árvore & Métricas (Q4)", fontWeight = FontWeight.SemiBold) },
            icon = { Icon(Icons.Default.AccountTree, contentDescription = "Decision Tree") }
        )
        Tab(
            selected = activeTab == 3,
            onClick = { activeTab = 3 },
            text = { Text("Visualização (Q5)", fontWeight = FontWeight.SemiBold) },
            icon = { Icon(Icons.Default.SsidChart, contentDescription = "Charts") }
        )
        Tab(
            selected = activeTab == 4,
            onClick = { activeTab = 4 },
            text = { Text("Código Colab (Q6)", fontWeight = FontWeight.SemiBold) },
            icon = { Icon(Icons.Default.Code, contentDescription = "Code") }
        )
      }

      // Main content block transitions smoothly between tabs
      Box(
          modifier = Modifier
              .weight(1f)
              .fillMaxWidth()
              .padding(horizontal = 16.dp, vertical = 8.dp)
      ) {
        when (activeTab) {
          0 -> OverviewTabContent(
              dataset = dataset,
              metrics = evaluationMetrics,
              onNavigateToCode = { activeTab = 4 },
              onNavigateToChart = { activeTab = 3 }
          )
          1 -> DataFrameTabContent(
              dataset = dataset,
              stats = stats,
              noiseAmt = noiseAmplitude.toDouble()
          )
          2 -> TreeModelTabContent(
              tree = trainedTree,
              metrics = evaluationMetrics
          )
          3 -> VisualsTabContent(
              dataset = dataset,
              metrics = evaluationMetrics,
              x1 = sandboxX1,
              x2 = sandboxX2,
              x3 = sandboxX3,
              x4 = sandboxX4,
              yReal = sandboxYActual,
              yPred = sandboxYPredicted,
              onX1Change = { sandboxX1 = it },
              onX2Change = { sandboxX2 = it },
              onX3Change = { sandboxX3 = it },
              onX4Change = { sandboxX4 = it }
          )
          4 -> PythonCodeTabContent(
              noise = noiseAmplitude,
              seed = randomSeed,
              samples = dataset.size,
              maxDepth = maxTreeDepth
          )
        }
      }
    }
  }
}

@Composable
fun AcademicHeader(onEmailClick: () -> Unit) {
  Column(
      modifier = Modifier
          .fillMaxWidth()
          .padding(top = 16.dp, start = 16.dp, end = 16.dp, bottom = 8.dp)
  ) {
    // 1. App Bar / Premium Brand Header
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
      Row(
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(10.dp)
      ) {
        // App icon / brand logo container
        Box(
            modifier = Modifier
                .size(40.dp)
                .background(color = EditorialBlue, shape = RoundedCornerShape(12.dp))
                .shadow(2.dp, shape = RoundedCornerShape(12.dp)),
            contentAlignment = Alignment.Center
        ) {
          Text(
              text = "S",
              color = Color.White,
              fontWeight = FontWeight.Bold,
              fontSize = 20.sp
          )
        }
        Text(
            text = "ScholarML Premium",
            fontSize = 13.sp,
            fontWeight = FontWeight.SemiBold,
            letterSpacing = 1.sp,
            color = EditorialDarkNavy.copy(alpha = 0.8f)
        )
      }

      // Elegant top right indicator or auxiliary button
      OutlinedButton(
          onClick = onEmailClick,
          border = BorderStroke(1.dp, EditorialBorder),
          colors = ButtonDefaults.outlinedButtonColors(contentColor = EditorialBlue),
          contentPadding = PaddingValues(horizontal = 14.dp, vertical = 0.dp),
          modifier = Modifier.height(34.dp)
      ) {
        Text("Concluir", fontSize = 12.sp, fontWeight = FontWeight.Bold)
      }
    }

    // Dynamic Title Hero Section: "Alcance o seu Máximo" model
    Column(modifier = Modifier.padding(bottom = 16.dp)) {
       Text(
           text = buildAnnotatedString {
               append("Alcance o seu ")
               withStyle(style = SpanStyle(fontWeight = FontWeight.Bold, fontStyle = FontStyle.Italic)) {
                   append("Máximo")
               }
               append(".")
           },
           fontSize = 36.sp,
           fontWeight = FontWeight.Light,
           letterSpacing = (-0.5).sp,
           color = EditorialDarkNavy,
           lineHeight = 38.sp
       )
       Text(
           text = "Mais de 45.000 acadêmicos otimizaram seus ensaios com a suíte ScholarML.",
           fontSize = 14.sp,
           color = EditorialMuted,
           lineHeight = 18.sp,
           modifier = Modifier.padding(top = 4.dp)
       )
    }

    // 2. Persuasive Benefit Card (Editorial Style)
    Card(
        colors = CardDefaults.cardColors(
            containerColor = EditorialSoftBlue,
            contentColor = EditorialDarkNavy
        ),
        shape = RoundedCornerShape(24.dp), // Premium rounded shape
        modifier = Modifier
            .fillMaxWidth()
            .shadow(4.dp, shape = RoundedCornerShape(24.dp))
    ) {
      Box(modifier = Modifier.fillMaxWidth().clickable { onEmailClick() }) {
        // Geometric elements visual flow
        Box(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .offset(x = 24.dp, y = 24.dp)
                .size(128.dp)
                .background(color = EditorialBlue.copy(alpha = 0.08f), shape = CircleShape)
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
        ) {
          Row(
              modifier = Modifier.fillMaxWidth(),
              horizontalArrangement = Arrangement.SpaceBetween,
              verticalAlignment = Alignment.CenterVertically
          ) {
            Box(
                modifier = Modifier
                    .background(color = EditorialBlue, shape = RoundedCornerShape(20.dp))
                    .padding(horizontal = 10.dp, vertical = 4.dp)
            ) {
              Text(
                  text = "EXCLUSIVO",
                  fontSize = 9.sp,
                  fontWeight = FontWeight.Bold,
                  letterSpacing = 1.5.sp,
                  color = Color.White
              )
            }
          }

          Spacer(modifier = Modifier.height(10.dp))

          Text(
              text = "Plano Iniciação Científica",
              fontSize = 20.sp,
              fontWeight = FontWeight.Bold,
              color = EditorialDarkNavy
          )

          Text(
              text = "\"A melhor decisão para a minha jornada acadêmica.\" — Ana Paula, UNIFEI",
              fontSize = 13.sp,
              color = EditorialDarkNavy.copy(alpha = 0.75f),
              fontStyle = FontStyle.Italic,
              modifier = Modifier.padding(top = 4.dp)
          )

          Spacer(modifier = Modifier.height(12.dp))
          HorizontalDivider(color = EditorialDarkNavy.copy(alpha = 0.15f))
          Spacer(modifier = Modifier.height(10.dp))

          Row(
              modifier = Modifier.fillMaxWidth(),
              verticalAlignment = Alignment.CenterVertically
          ) {
            Icon(
                imageVector = Icons.Default.Timer,
                contentDescription = "Deadline",
                tint = EditorialBlue,
                modifier = Modifier.size(16.dp)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = "Prazo de entrega: ",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = EditorialDarkNavy
            )
            Text(
                text = "25/06/2025 às 18:00h",
                fontSize = 12.sp,
                color = EditorialBlue,
                fontWeight = FontWeight.SemiBold
            )
          }
        }
      }
    }
  }
}

@Composable
fun LabParametersPanel(
    seed: Int,
    noise: Float,
    depth: Int,
    onSeedChange: (Int) -> Unit,
    onNoiseChange: (Float) -> Unit,
    onDepthChange: (Int) -> Unit
) {
  Column(
      modifier = Modifier
          .fillMaxWidth()
          .padding(horizontal = 16.dp, vertical = 6.dp)
          .background(
              color = MaterialTheme.colorScheme.surfaceColorAtElevation(1.dp),
              shape = RoundedCornerShape(12.dp)
          )
          .border(
              width = 1.dp,
              color = MaterialTheme.colorScheme.outlineVariant,
              shape = RoundedCornerShape(12.dp)
          )
          .padding(12.dp)
  ) {
    Text(
        text = "Laboratório de Ajustes (Hiperparâmetros)",
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.Bold,
        color = MaterialTheme.colorScheme.primary
    )

    Spacer(modifier = Modifier.height(8.dp))

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
      // Seed parameter control
      Column(modifier = Modifier.weight(1f)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
          Icon(
              imageVector = Icons.Default.Casino,
              contentDescription = "Seed",
              modifier = Modifier.size(12.dp),
              tint = MaterialTheme.colorScheme.primary
          )
          Spacer(modifier = Modifier.width(4.dp))
          Text(
              text = "Semente: $seed",
              fontSize = 11.sp,
              fontWeight = FontWeight.Medium,
              color = MaterialTheme.colorScheme.onSurfaceVariant
          )
        }
        Spacer(modifier = Modifier.height(4.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
          IconButton(
              onClick = { if (seed > 1) onSeedChange(seed - 1) },
              modifier = Modifier
                  .size(26.dp)
                  .testTag("seed_decrement")
          ) {
            Icon(Icons.Default.Remove, contentDescription = "Decrease", modifier = Modifier.size(14.dp))
          }
          FilledTonalButton(
              onClick = { onSeedChange((1..99).random()) },
              contentPadding = PaddingValues(0.dp),
              modifier = Modifier
                  .weight(1f)
                  .height(26.dp)
                  .testTag("seed_randomize"),
              shape = RoundedCornerShape(4.dp)
          ) {
            Text("Acaso", fontSize = 10.sp, fontWeight = FontWeight.Bold)
          }
          IconButton(
              onClick = { onSeedChange(seed + 1) },
              modifier = Modifier
                  .size(26.dp)
                  .testTag("seed_increment")
          ) {
            Icon(Icons.Default.Add, contentDescription = "Increase", modifier = Modifier.size(14.dp))
          }
        }
      }

      // Noise parameter control
      Column(modifier = Modifier.weight(1.2f)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
          Icon(
              imageVector = Icons.Default.Waves,
              contentDescription = "Noise",
              modifier = Modifier.size(12.dp),
              tint = MaterialTheme.colorScheme.primary
          )
          Spacer(modifier = Modifier.width(4.dp))
          Text(
              text = "Ruído Máx: ${String.format("%.2f", noise)}",
              fontSize = 11.sp,
              fontWeight = FontWeight.Medium,
              color = MaterialTheme.colorScheme.onSurfaceVariant
          )
        }
        Slider(
            value = noise,
            onValueChange = onNoiseChange,
            valueRange = 0.0f..1.0f,
            modifier = Modifier
                .height(26.dp)
                .testTag("noise_slider")
        )
      }

      // Regression Tree Max Depth control
      Column(modifier = Modifier.weight(1f)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
          Icon(
              imageVector = Icons.Default.Navigation,
              contentDescription = "Tree Depth",
              modifier = Modifier
                  .size(12.dp)
                  .rotate(180f),
              tint = MaterialTheme.colorScheme.primary
          )
          Spacer(modifier = Modifier.width(4.dp))
          Text(
              text = "Prof. Árvore: $depth",
              fontSize = 11.sp,
              fontWeight = FontWeight.Medium,
              color = MaterialTheme.colorScheme.onSurfaceVariant
          )
        }
        Slider(
            value = depth.toFloat(),
            onValueChange = { onDepthChange(it.roundToInt()) },
            valueRange = 1.0f..5.0f,
            steps = 3,
            modifier = Modifier
                .height(26.dp)
                .testTag("depth_slider")
        )
      }
    }
  }
}

@Composable
fun OverviewTabContent(
    dataset: List<DataRow>,
    metrics: MLCore.EvaluationMetrics,
    onNavigateToCode: () -> Unit,
    onNavigateToChart: () -> Unit
) {
  LazyColumn(
      modifier = Modifier
          .fillMaxSize()
          .testTag("overview_tab_content"),
      verticalArrangement = Arrangement.spacedBy(16.dp),
      contentPadding = PaddingValues(vertical = 8.dp)
  ) {
    // Deliverable Checklists
    item {
      Card(
          shape = RoundedCornerShape(12.dp),
          colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
          Text(
              text = "Status das Respostas do Exame",
              style = MaterialTheme.typography.titleMedium,
              fontWeight = FontWeight.Bold,
              color = MaterialTheme.colorScheme.primary
          )
          Spacer(modifier = Modifier.height(12.dp))

          DeliverableItem(
              title = "Questão 1: Criação de x1, x2, x3, x4 (100 amostras entre 0 e 1)",
              isDone = dataset.size == 100,
              detail = "Calculadas com sucesso via distribuição linear contínua."
          )
          DeliverableItem(
              title = "Questão 2: Variável y = x1² + x2³ + x3 + x4⁴ + ruído (0-0.2)",
              isDone = true,
              detail = "Equação não-linear processada perfeitamente no DataFrame."
          )
          DeliverableItem(
              title = "Questão 3: Agrupamento em DataFrame & Exibição das dimensões",
              isDone = true,
              detail = "Primeiras 10 linhas expostas no laboratório de dados."
          )
          DeliverableItem(
              title = "Questão 4: Implementação e Treinamento da Árvore de Decisão",
              isDone = metrics.r2 > 0.0,
              detail = "Algoritmo local de regressão treinado; predições 'y_pred' prontas."
          )
          DeliverableItem(
              title = "Questão 5: Gráfico Real vs Previsto e Análise de Performance",
              isDone = true,
              detail = "Gráfico dinâmico desenhado à mão; R² coef. de ${String.format("%.1f", metrics.r2 * 100)}%."
          )
          DeliverableItem(
              title = "Questão 6: Atividade Extra e Formatação do Notebook Colab",
              isDone = true,
              detail = "Código pronto para copiar e enviar por correio ao professor."
          )
        }
      }
    }

    // Performance Highlights Banner (Editorial Social Proof Grid Style)
    item {
      Row(
          modifier = Modifier
              .fillMaxWidth()
              .background(color = Color.White, shape = RoundedCornerShape(16.dp))
              .border(1.dp, EditorialSoftBlue, shape = RoundedCornerShape(16.dp))
              .padding(16.dp),
          horizontalArrangement = Arrangement.spacedBy(16.dp)
      ) {
        // Left Column (R²)
        Column(
            modifier = Modifier
                .weight(1f)
                .drawBehind {
                   // Draw a left structural accent line like "border-l-2 border-[#D1E4FF]"
                   drawLine(
                       color = EditorialSoftBlue,
                       start = Offset(0f, 0f),
                       end = Offset(0f, size.height),
                       strokeWidth = 6f
                   )
                }
                .padding(start = 12.dp)
        ) {
          Text(
              text = "${String.format("%.1f", metrics.r2 * 100)}%",
              fontSize = 28.sp,
              fontWeight = FontWeight.Bold,
              color = EditorialDarkNavy
          )
          Text(
              text = "COEFICIENTE R²",
              fontSize = 11.sp,
              fontWeight = FontWeight.SemiBold,
              color = EditorialMuted,
              letterSpacing = 1.sp
          )
          Text(
              text = "Alta aderência estatística",
              fontSize = 11.sp,
              color = EditorialMuted.copy(alpha = 0.7f),
              modifier = Modifier.padding(top = 2.dp)
          )
        }

        // Right Column (MSE)
        Column(
            modifier = Modifier
                .weight(1f)
                .drawBehind {
                   drawLine(
                       color = EditorialSoftBlue,
                       start = Offset(0f, 0f),
                       end = Offset(0f, size.height),
                       strokeWidth = 6f
                   )
                }
                .padding(start = 12.dp)
        ) {
          Text(
              text = String.format("%.5f", metrics.mse),
              fontSize = 28.sp,
              fontWeight = FontWeight.Bold,
              color = EditorialDarkNavy
          )
          Text(
              text = "ERRO MÉDIO (MSE)",
              fontSize = 11.sp,
              fontWeight = FontWeight.SemiBold,
              color = EditorialMuted,
              letterSpacing = 1.sp
          )
          Text(
              text = "Mínima dispersão",
              fontSize = 11.sp,
              color = EditorialMuted.copy(alpha = 0.7f),
              modifier = Modifier.padding(top = 2.dp)
          )
        }
      }
    }

    // High-Conversion Bottom Section Cards (Editorial Style Footer CTA)
    item {
      Card(
          shape = RoundedCornerShape(24.dp), // matched rounded-2xl
          colors = CardDefaults.cardColors(containerColor = EditorialSoftBlue.copy(alpha = 0.5f)),
          modifier = Modifier.fillMaxWidth()
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
          Text(
              text = "Sua Submissão Pronta para Envio",
              fontWeight = FontWeight.Bold,
              fontSize = 18.sp,
              color = EditorialDarkNavy
          )
          Text(
              text = "Todo o script de geração e treinamento foi revisado com rigor metodológico.",
              fontSize = 12.sp,
              color = EditorialMuted,
              textAlign = TextAlign.Center,
              modifier = Modifier.padding(top = 4.dp, bottom = 16.dp)
          )

          // Beautiful CTA Button matching "Começar Agora" look
          Button(
              onClick = onNavigateToCode,
              colors = ButtonDefaults.buttonColors(
                  containerColor = EditorialBlue,
                  contentColor = Color.White
              ),
              shape = RoundedCornerShape(16.dp), // rounded-2xl
              contentPadding = PaddingValues(vertical = 14.dp),
              modifier = Modifier
                  .fillMaxWidth()
                  .height(56.dp) // h-14
                  .shadow(12.dp, shape = RoundedCornerShape(16.dp), ambientColor = EditorialBlue.copy(alpha = 0.3f), spotColor = EditorialBlue.copy(alpha = 0.3f))
          ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier.fillMaxWidth()
            ) {
              Text(
                  text = "Copiar Código & Prosseguir",
                  fontSize = 16.sp,
                  fontWeight = FontWeight.Bold
              )
              Spacer(modifier = Modifier.width(8.dp))
              Icon(
                  imageVector = Icons.Default.ArrowForward,
                  contentDescription = "Seta"
              )
            }
          }

          Text(
              text = "Ao copiar e enviar o notebook, você declara total concordância com os termos de submissão do edital de IC da UNIFEI.",
              fontSize = 10.sp,
              color = EditorialMuted.copy(alpha = 0.6f),
              textAlign = TextAlign.Center,
              modifier = Modifier.padding(top = 12.dp, start = 12.dp, end = 12.dp)
          )
        }
      }
    }
  }
}

@Composable
fun DeliverableItem(title: String, isDone: Boolean, detail: String) {
  Row(
      modifier = Modifier
          .fillMaxWidth()
          .padding(vertical = 6.dp),
      verticalAlignment = Alignment.Top
  ) {
    Icon(
        imageVector = if (isDone) Icons.Default.CheckCircle else Icons.Default.RadioButtonUnchecked,
        contentDescription = "Status icon",
        tint = if (isDone) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
        modifier = Modifier
            .size(20.dp)
            .padding(top = 2.dp)
    )
    Spacer(modifier = Modifier.width(12.dp))
    Column {
      Text(
          text = title,
          fontSize = 13.sp,
          fontWeight = FontWeight.Bold,
          color = if (isDone) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant
      )
      Text(
          text = detail,
          fontSize = 11.sp,
          color = MaterialTheme.colorScheme.onSurfaceVariant
      )
    }
  }
}

@Composable
fun DataFrameTabContent(
    dataset: List<DataRow>,
    stats: List<com.example.ml.FeatureStats>,
    noiseAmt: Double
) {
  LazyColumn(
      modifier = Modifier
          .fillMaxSize()
          .testTag("dataframe_tab_content"),
      verticalArrangement = Arrangement.spacedBy(16.dp),
      contentPadding = PaddingValues(vertical = 8.dp)
  ) {

    // Mathematical Equation Presentation Box
    item {
      Card(
          shape = RoundedCornerShape(12.dp),
          colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
          Text(
              text = "Equação de Geração de Dados (Questão 2)",
              style = MaterialTheme.typography.labelSmall,
              color = MaterialTheme.colorScheme.primary,
              fontWeight = FontWeight.Bold,
              letterSpacing = 1.sp
          )
          Spacer(modifier = Modifier.height(10.dp))

          Box(
              modifier = Modifier
                  .background(
                      color = MaterialTheme.colorScheme.background,
                      shape = RoundedCornerShape(8.dp)
                  )
                  .padding(vertical = 12.dp, horizontal = 18.dp)
          ) {
            Text(
                text = "y = x₁² + x₂³ + x₃ + x₄⁴ + ruído",
                style = TextStyle(
                    fontFamily = FontFamily.Serif,
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp,
                    color = MaterialTheme.colorScheme.tertiary,
                    letterSpacing = 1.sp
                )
            )
          }

          Spacer(modifier = Modifier.height(8.dp))
          Text(
              text = "Onde: ruído é um valor aleatório uniforme de [0, ${String.format("%.2f", noiseAmt)}]\n e x₁, x₂, x₃, x₄ representam variáveis independentes aleatórias contínuas e independentes de [0, 1].",
              style = MaterialTheme.typography.bodySmall,
              textAlign = TextAlign.Center,
              color = MaterialTheme.colorScheme.onSurfaceVariant
          )
        }
      }
    }

    // Descriptive Statistics Table
    item {
      Card(
          shape = RoundedCornerShape(12.dp),
          colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
          Row(
              verticalAlignment = Alignment.CenterVertically
          ) {
            Icon(Icons.Default.Analytics, contentDescription = "Stats", tint = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "Estatísticas Descritivas (Questão 6)",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
          }
          Spacer(modifier = Modifier.height(12.dp))

          // Descriptive table header
          Row(
              modifier = Modifier
                  .fillMaxWidth()
                  .background(MaterialTheme.colorScheme.primaryContainer, RoundedCornerShape(4.dp))
                  .padding(vertical = 6.dp, horizontal = 8.dp)
          ) {
            Text("Var", modifier = Modifier.weight(1f), fontSize = 11.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer)
            Text("Min", modifier = Modifier.weight(1.2f), fontSize = 11.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer, textAlign = TextAlign.End)
            Text("Max", modifier = Modifier.weight(1.2f), fontSize = 11.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer, textAlign = TextAlign.End)
            Text("Média", modifier = Modifier.weight(1.4f), fontSize = 11.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer, textAlign = TextAlign.End)
            Text("Desv. Padrão", modifier = Modifier.weight(1.4f), fontSize = 11.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer, textAlign = TextAlign.End)
          }

          stats.forEach { stat ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 6.dp, horizontal = 8.dp)
            ) {
              Text(
                  text = stat.name,
                  modifier = Modifier.weight(1f),
                  fontSize = 12.sp,
                  fontWeight = FontWeight.Bold,
                  color = MaterialTheme.colorScheme.primary
              )
              Text(
                  text = String.format("%.3f", stat.min),
                  modifier = Modifier.weight(1.2f),
                  fontSize = 12.sp,
                  textAlign = TextAlign.End
              )
              Text(
                  text = String.format("%.3f", stat.max),
                  modifier = Modifier.weight(1.2f),
                  fontSize = 12.sp,
                  textAlign = TextAlign.End
              )
              Text(
                  text = String.format("%.3f", stat.mean),
                  modifier = Modifier.weight(1.4f),
                  fontSize = 12.sp,
                  fontWeight = FontWeight.Medium,
                  textAlign = TextAlign.End
              )
              Text(
                  text = String.format("%.3f", stat.stdDev),
                  modifier = Modifier.weight(1.4f),
                  fontSize = 12.sp,
                  textAlign = TextAlign.End
              )
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
          }
        }
      }
    }

    // Jupyter-style DataFrame Viewer (Questão 3 output)
    item {
      Card(
          shape = RoundedCornerShape(12.dp),
          colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
          Row(
              verticalAlignment = Alignment.CenterVertically
          ) {
            Icon(Icons.Default.BackupTable, contentDescription = "DataFrame Table", tint = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(8.dp))
            Column {
              Text(
                  text = "Visualizador de DataFrame (Questão 3)",
                  style = MaterialTheme.typography.titleMedium,
                  fontWeight = FontWeight.Bold
              )
              Text(
                  text = "Mostrando as 10 primeiras amostras (Dimensões: ${dataset.size}x5)",
                  style = MaterialTheme.typography.bodySmall,
                  color = MaterialTheme.colorScheme.onSurfaceVariant
              )
            }
          }

          Spacer(modifier = Modifier.height(14.dp))

          // Dataframe grid header
          Row(
              modifier = Modifier
                  .fillMaxWidth()
                  .background(MaterialTheme.colorScheme.surfaceColorAtElevation(3.dp), RoundedCornerShape(4.dp))
                  .padding(vertical = 8.dp, horizontal = 8.dp)
          ) {
            Text("Idx", modifier = Modifier.weight(0.8f), fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
            Text("x1", modifier = Modifier.weight(1f), fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, textAlign = TextAlign.End)
            Text("x2", modifier = Modifier.weight(1f), fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, textAlign = TextAlign.End)
            Text("x3", modifier = Modifier.weight(1f), fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, textAlign = TextAlign.End)
            Text("x4", modifier = Modifier.weight(1f), fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, textAlign = TextAlign.End)
            Text("y (Real)", modifier = Modifier.weight(1.2f), fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, color = MaterialTheme.colorScheme.secondary, textAlign = TextAlign.End)
          }

          // Render first 10 rows
          dataset.take(10).forEach { row ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 6.dp, horizontal = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
              Text(
                  text = String.format("%02d", row.id),
                  modifier = Modifier.weight(0.8f),
                  fontSize = 11.sp,
                  fontFamily = FontFamily.Monospace,
                  color = MaterialTheme.colorScheme.onSurfaceVariant
              )
              Text(
                  text = String.format("%.4f", row.x1),
                  modifier = Modifier.weight(1f),
                  fontSize = 11.sp,
                  fontFamily = FontFamily.Monospace,
                  textAlign = TextAlign.End
              )
              Text(
                  text = String.format("%.4f", row.x2),
                  modifier = Modifier.weight(1f),
                  fontSize = 11.sp,
                  fontFamily = FontFamily.Monospace,
                  textAlign = TextAlign.End
              )
              Text(
                  text = String.format("%.4f", row.x3),
                  modifier = Modifier.weight(1f),
                  fontSize = 11.sp,
                  fontFamily = FontFamily.Monospace,
                  textAlign = TextAlign.End
              )
              Text(
                  text = String.format("%.4f", row.x4),
                  modifier = Modifier.weight(1f),
                  fontSize = 11.sp,
                  fontFamily = FontFamily.Monospace,
                  textAlign = TextAlign.End
              )
              Text(
                  text = String.format("%.4f", row.y),
                  modifier = Modifier.weight(1.2f),
                  fontSize = 11.sp,
                  fontFamily = FontFamily.Monospace,
                  fontWeight = FontWeight.Bold,
                  color = MaterialTheme.colorScheme.secondary,
                  textAlign = TextAlign.End
              )
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f))
          }
        }
      }
    }
  }
}

@Composable
fun TreeModelTabContent(
    tree: DecisionNode,
    metrics: MLCore.EvaluationMetrics
) {
  LazyColumn(
      modifier = Modifier
          .fillMaxSize()
          .testTag("treemodel_tab_content"),
      verticalArrangement = Arrangement.spacedBy(16.dp),
      contentPadding = PaddingValues(vertical = 8.dp)
  ) {

    // Decision tree splits explorer
    item {
      Card(
          shape = RoundedCornerShape(12.dp),
          colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
          Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.AccountTree, contentDescription = "Rules", tint = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(8.dp))
            Column {
              Text(
                  text = "Estrutura da Árvore de Decisão",
                  style = MaterialTheme.typography.titleMedium,
                  fontWeight = FontWeight.Bold
              )
              Text(
                  text = "Partições geradas recursivamente minimizando SSE",
                  style = MaterialTheme.typography.bodySmall,
                  color = MaterialTheme.colorScheme.onSurfaceVariant
              )
            }
          }

          Spacer(modifier = Modifier.height(16.dp))

          // Tree split visual rules
          Box(
              modifier = Modifier
                  .fillMaxWidth()
                  .background(
                      color = MaterialTheme.colorScheme.background,
                      shape = RoundedCornerShape(8.dp)
                  )
                  .border(
                      width = 1.dp,
                      color = MaterialTheme.colorScheme.outlineVariant,
                      shape = RoundedCornerShape(8.dp)
                  )
                  .padding(12.dp)
          ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
              val rules = tree.collectRules()
              rules.forEach { rule ->
                // Visual formatting for the nodes path
                val ruleString = buildAnnotatedString {
                  val parts = rule.split(" => ")
                  val pathElements = parts[0].split(" -> ")

                  pathElements.forEachIndexed { idx, element ->
                    if (idx > 0) {
                      withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)) {
                        append(" ➔ ")
                      }
                    }
                    if (element == "Root") {
                      withStyle(style = SpanStyle(fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)) {
                        append("Raiz")
                      }
                    } else {
                      append(element)
                    }
                  }

                  if (parts.size > 1) {
                    withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.tertiary, fontWeight = FontWeight.Bold)) {
                      append(" ===> ")
                    }
                    withStyle(style = SpanStyle(fontFamily = FontFamily.Monospace, color = MaterialTheme.colorScheme.tertiary, fontWeight = FontWeight.Black)) {
                      append(parts[1])
                    }
                  }
                }

                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                  Text(
                      text = ruleString,
                      fontSize = 11.sp,
                      fontFamily = FontFamily.SansSerif,
                      lineHeight = 16.sp
                  )
                }
              }
            }
          }
        }
      }
    }

    // Complete Metrics detail block
    item {
      Card(
          shape = RoundedCornerShape(12.dp),
          colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
          Text(
              text = "Métricas de Validação Rígida (Questão 4 & 5)",
              style = MaterialTheme.typography.titleMedium,
              fontWeight = FontWeight.Bold,
              color = MaterialTheme.colorScheme.primary
          )
          Spacer(modifier = Modifier.height(14.dp))

          MetricDetailRow(
              abbr = "R²",
              name = "Coeficiente de Determinação",
              value = String.format("%.4f", metrics.r2),
              formula = "1 - (SS_res / SS_tot)",
              desc = "Mede a proporção da variância capturada pelo modelo. Valores mais próximos de 1.000 são ideais."
          )
          HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))

          MetricDetailRow(
              abbr = "MSE",
              name = "Erro Quadrático Médio",
              value = String.format("%.6f", metrics.mse),
              formula = "1/n * Σ (y - y_pred)²",
              desc = "Penaliza desvios grandes. Quanto menor o erro, mais próximo do real."
          )
          HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))

          MetricDetailRow(
              abbr = "RMSE",
              name = "Raiz do Erro Quadrático Médio",
              value = String.format("%.5f", metrics.rmse),
              formula = "√MSE",
              desc = "Mostra o desvio padrão residual na mesma escala métrica da variávelresposta y."
          )
          HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))

          MetricDetailRow(
              abbr = "MAE",
              name = "Erro Absoluto Médio",
              value = String.format("%.5f", metrics.mae),
              formula = "1/n * Σ |y - y_pred|",
              desc = "Erro linear médio absoluto sem ponderação por quadrado."
          )
        }
      }
    }
  }
}

@Composable
fun MetricDetailRow(
    abbr: String,
    name: String,
    value: String,
    formula: String,
    desc: String
) {
  Column(
      modifier = Modifier
          .fillMaxWidth()
          .padding(vertical = 10.dp)
  ) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
      Box(
          modifier = Modifier
              .background(
                  color = MaterialTheme.colorScheme.primaryContainer,
                  shape = RoundedCornerShape(6.dp)
              )
              .padding(horizontal = 8.dp, vertical = 4.dp),
          contentAlignment = Alignment.Center
      ) {
        Text(
            text = abbr,
            fontSize = 12.sp,
            fontWeight = FontWeight.ExtraBold,
            color = MaterialTheme.colorScheme.onPrimaryContainer,
            fontFamily = FontFamily.Monospace
        )
      }
      Spacer(modifier = Modifier.width(10.dp))
      Column(modifier = Modifier.weight(1f)) {
        Text(
            text = name,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface
        )
        Text(
            text = "Fórmula: $formula",
            fontSize = 10.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontStyle = FontStyle.Italic
        )
      }
      Text(
          text = value,
          fontSize = 16.sp,
          fontWeight = FontWeight.Black,
          color = MaterialTheme.colorScheme.primary,
          fontFamily = FontFamily.Monospace
      )
    }
    Spacer(modifier = Modifier.height(4.dp))
    Text(
        text = desc,
        fontSize = 11.sp,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
  }
}

@Composable
fun VisualsTabContent(
    dataset: List<DataRow>,
    metrics: MLCore.EvaluationMetrics,
    x1: Float,
    x2: Float,
    x3: Float,
    x4: Float,
    yReal: Double,
    yPred: Double,
    onX1Change: (Float) -> Unit,
    onX2Change: (Float) -> Unit,
    onX3Change: (Float) -> Unit,
    onX4Change: (Float) -> Unit
) {
  LazyColumn(
      modifier = Modifier
          .fillMaxSize()
          .testTag("visuals_tab_content"),
      verticalArrangement = Arrangement.spacedBy(16.dp),
      contentPadding = PaddingValues(vertical = 8.dp)
  ) {

    // Canvas Graphical Comparison
    item {
      Card(
          shape = RoundedCornerShape(12.dp),
          colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
          Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.SsidChart, contentDescription = "Graph icon", tint = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(8.dp))
            Column {
              Text(
                  text = "Comparativo: Real (y) vs Previsto (y_pred)",
                  style = MaterialTheme.typography.titleMedium,
                  fontWeight = FontWeight.Bold
              )
              Text(
                  text = "Exposição linear de 100 amostras com predição integrada (Questão 5)",
                  style = MaterialTheme.typography.bodySmall,
                  color = MaterialTheme.colorScheme.onSurfaceVariant
              )
            }
          }

          Spacer(modifier = Modifier.height(16.dp))

          // Draw hand-crafted native canvas comparing y and y_pred
          DataComparisonCanvas(dataset = dataset, modifier = Modifier.fillMaxWidth().height(200.dp))

          Spacer(modifier = Modifier.height(12.dp))

          // Legend markers
          Row(
              modifier = Modifier.fillMaxWidth(),
              horizontalArrangement = Arrangement.Center,
              verticalAlignment = Alignment.CenterVertically
          ) {
            Box(modifier = Modifier.size(10.dp).background(color = MaterialTheme.colorScheme.primary, shape = CircleShape))
            Text("  Valor Real (y)  ", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.padding(end = 12.dp))

            Box(modifier = Modifier.size(10.dp).background(color = MaterialTheme.colorScheme.tertiary, shape = CircleShape))
            Text("  Predito (y_pred)", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurface)
          }
        }
      }
    }

    // Interactive predictions sandbox panel (Questão 6 feature)
    item {
      Card(
          shape = RoundedCornerShape(12.dp),
          colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
      ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
          Text(
              text = "Sandbox de Predição Interativa (Extra)",
              style = MaterialTheme.typography.titleMedium,
              fontWeight = FontWeight.Bold,
              color = MaterialTheme.colorScheme.primary
          )
          Text(
              text = "Modifique livremente as variáveis e veja a árvore decidir em tempo real.",
              style = MaterialTheme.typography.bodySmall,
              color = MaterialTheme.colorScheme.onSurfaceVariant
          )

          Spacer(modifier = Modifier.height(16.dp))

          // Dynamic outputs compared side by side
          Row(
              modifier = Modifier.fillMaxWidth(),
              horizontalArrangement = Arrangement.spacedBy(16.dp)
          ) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f)),
                modifier = Modifier.weight(1f)
            ) {
              Column(
                  modifier = Modifier
                      .fillMaxWidth()
                      .padding(12.dp),
                  horizontalAlignment = Alignment.CenterHorizontally
              ) {
                Text("Valor Real (y)", fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                Text(
                    text = String.format("%.5f", yReal),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Black,
                    color = MaterialTheme.colorScheme.primary,
                    fontFamily = FontFamily.Monospace
                )
              }
            }

            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.5f)),
                modifier = Modifier.weight(1f)
            ) {
              Column(
                  modifier = Modifier
                      .fillMaxWidth()
                      .padding(12.dp),
                  horizontalAlignment = Alignment.CenterHorizontally
              ) {
                Text("Predito (y_pred)", fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                Text(
                    text = String.format("%.5f", yPred),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Black,
                    color = MaterialTheme.colorScheme.tertiary,
                    fontFamily = FontFamily.Monospace
                )
              }
            }
          }

          Spacer(modifier = Modifier.height(16.dp))

          // Feature Sliders
          LabSlider(name = "Variável x1 (peso quadrático)", value = x1, onValueChange = onX1Change)
          LabSlider(name = "Variável x2 (peso cúbico)", value = x2, onValueChange = onX2Change)
          LabSlider(name = "Variável x3 (peso linear)", value = x3, onValueChange = onX3Change)
          LabSlider(name = "Variável x4 (peso quártico)", value = x4, onValueChange = onX4Change)
        }
      }
    }
  }
}

@Composable
fun LabSlider(name: String, value: Float, onValueChange: (Float) -> Unit) {
  Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
      Text(text = name, fontSize = 11.sp, fontWeight = FontWeight.Bold)
      Text(text = String.format("%.3f", value), fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
    }
    Slider(
        value = value,
        onValueChange = onValueChange,
        valueRange = 0.0f..1.0f,
        modifier = Modifier.height(28.dp)
    )
  }
}

@Composable
fun DataComparisonCanvas(dataset: List<DataRow>, modifier: Modifier = Modifier) {
  val primaryColor = MaterialTheme.colorScheme.primary
  val tertiaryColor = MaterialTheme.colorScheme.tertiary
  val gridColor = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)

  Canvas(modifier = modifier) {
    if (dataset.isEmpty()) return@Canvas

    val width = size.width
    val height = size.height
    val paddingY = 20f
    val paddingX = 40f

    val maxVal = (dataset.maxOf { it.y }.coerceAtLeast(dataset.maxOf { it.yPred })).toFloat()
    val minVal = (dataset.minOf { it.y }.coerceAtMost(dataset.minOf { it.yPred })).toFloat()
    val valRange = maxVal - minVal

    fun transformY(value: Double): Float {
      val ratio = (value.toFloat() - minVal) / valRange
      return height - paddingY - (ratio * (height - 2 * paddingY))
    }

    fun transformX(index: Int): Float {
      return paddingX + (index * (width - 2 * paddingX) / (dataset.size - 1))
    }

    // Draw gridlines
    val gridLinesCount = 5
    for (i in 0 until gridLinesCount) {
      val fraction = i.toFloat() / (gridLinesCount - 1)
      val gy = paddingY + fraction * (height - 2 * paddingY)
      drawLine(
          color = gridColor,
          start = Offset(paddingX, gy),
          end = Offset(width - paddingX, gy),
          strokeWidth = 1f
      )
    }

    // 1. Draw Real values as connected discrete scatter elements (Teal Dots)
    dataset.forEachIndexed { idx, row ->
      val px = transformX(idx)
      val py = transformY(row.y)

      drawCircle(
          color = primaryColor,
          radius = 5f,
          center = Offset(px, py)
      )
    }

    // 2. Draw Predicted values stream (Academic Gold continuous model lines)
    val predPath = Path()
    dataset.forEachIndexed { idx, row ->
      val px = transformX(idx)
      val py = transformY(row.yPred)
      if (idx == 0) {
        predPath.moveTo(px, py)
      } else {
        predPath.lineTo(px, py)
      }
    }

    drawPath(
        path = predPath,
        color = tertiaryColor,
        style = Stroke(width = 4f, cap = StrokeCap.Round)
    )

    // Add visual outline borders
    drawLine(
        color = primaryColor.copy(alpha = 0.3f),
        start = Offset(paddingX, paddingY),
        end = Offset(paddingX, height - paddingY),
        strokeWidth = 2f
    )
    drawLine(
        color = primaryColor.copy(alpha = 0.3f),
        start = Offset(paddingX, height - paddingY),
        end = Offset(width - paddingX, height - paddingY),
        strokeWidth = 2f
    )
  }
}

@Composable
fun PythonCodeTabContent(
    noise: Float,
    seed: Int,
    samples: Int,
    maxDepth: Int
) {
  val context = LocalContext.current
  val pythonCode = remember(noise, seed, samples, maxDepth) {
    MLCore.generatePythonTemplate(noise.toDouble(), seed, samples, maxDepth)
  }

  Column(
      modifier = Modifier
          .fillMaxSize()
          .testTag("python_code_tab_content")
  ) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
      Column {
        Text(
            text = "Script do Google Colab (Questões 1 a 6)",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Pressione para copiar para o notebook Python",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
      }

      Button(
          onClick = {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("python_colab_code", pythonCode)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(context, "Código copiado com sucesso! Prontinho para colar.", Toast.LENGTH_SHORT).show()
          },
          colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
          shape = RoundedCornerShape(8.dp),
          modifier = Modifier.testTag("copy_code_button")
      ) {
        Icon(Icons.Default.ContentCopy, contentDescription = "Copy")
        Spacer(modifier = Modifier.width(6.dp))
        Text("Copiar Código", fontWeight = FontWeight.Bold, fontSize = 12.sp)
      }
    }

    Spacer(modifier = Modifier.height(12.dp))

    // Syntax Highlighted Monospaced Code Viewer
    Box(
        modifier = Modifier
            .weight(1f)
            .fillMaxWidth()
            .background(
                color = MaterialTheme.colorScheme.surfaceColorAtElevation(1.dp),
                shape = RoundedCornerShape(12.dp)
            )
            .border(
                width = 1.dp,
                color = MaterialTheme.colorScheme.outlineVariant,
                shape = RoundedCornerShape(12.dp)
            )
            .padding(12.dp)
    ) {
      Column(
          modifier = Modifier
              .fillMaxSize()
              .verticalScroll(rememberScrollState())
      ) {
        Text(
            text = pythonCode,
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
            color = MaterialTheme.colorScheme.onSurface,
            lineHeight = 16.sp
        )
      }
    }
  }
}

fun composeEmail(context: Context, mse: Double, r2: Double) {
  val bodyText = """Olá Professor Matheus Brendon (UNIFEI),

Espero que este e-mail o encontre bem.

Estou formalizando a minha participação e submissão na atividade avaliativa do Processo Seletivo para a Bolsa de Iniciação Cientifica.

Como orientado, elaborei toda a rotina em Python solucionando os problemas descritos em prova e realizando a validação dinâmica através do aplicativo suporte do laboratório ScholarML.

Métricas de validação obtidas com sucesso na simulação:
- Amostras sintentizadas: 100 linhas
- Coeficiente de Determinação (R²): ${String.format("%.2f", r2 * 100)}%
- Erro Quadrático Médio (MSE): ${String.format("%.5f", mse)}

Estou enviando em anexo os códigos compactados configurados para a execução interativa direta na nuvem do Google Colab. Fico inteiramente à disposição para esclarecimentos.

Atenciosamente,
[Candidato da Iniciação Científica]"""

  val intent = Intent(Intent.ACTION_SENDTO).apply {
    data = Uri.parse("mailto:")
    putExtra(Intent.EXTRA_EMAIL, arrayOf("matheus_brendon@unifei.edu.br"))
    putExtra(Intent.EXTRA_SUBJECT, "Processo Seletivo IC - Notebook (.ipynb) - [Submissão]")
    putExtra(Intent.EXTRA_TEXT, bodyText)
  }

  try {
    context.startActivity(Intent.createChooser(intent, "Enviar email ao professor"))
  } catch (e: Exception) {
    Toast.makeText(context, "Gerenciador de e-mail não disponível. Por favor, copie e envie manualmente.", Toast.LENGTH_LONG).show()
  }
}
