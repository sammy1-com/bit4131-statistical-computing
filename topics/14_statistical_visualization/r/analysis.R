# Topic 14: Statistical data visualization (ggplot2, lattice, plotly)
source("common/helpers.R"); start_topic("14_statistical_visualization")
need("ggplot2", "lattice")

df <- load_data("students.csv"); df$method <- factor(df$method)
sales <- load_data("monthly_sales.csv"); sales$date <- as.Date(sales$date)

h("WORKED EXAMPLE: choosing the right chart")
cat("One numeric variable        -> histogram / density / boxplot\n")
cat("Numeric by category         -> boxplot / violin\n")
cat("Two numeric variables       -> scatter plot (+ regression line)\n")
cat("Numeric over time           -> line graph\n")
cat("Counts of a category        -> bar chart\n")

h("DATASET: statistical charts")
p1 <- ggplot(df, aes(exam_score)) + geom_histogram(bins = 20, fill = "steelblue", colour = "white") +
  labs(title = "Histogram: exam score", x = "Score", y = "Students")
p2 <- ggplot(df, aes(method, exam_score, fill = method)) + geom_boxplot(alpha = .7) + labs(title = "Boxplot: score by method")
p3 <- ggplot(df, aes(method, exam_score, fill = method)) + geom_violin(alpha = .7) + geom_boxplot(width = .1, fill = "white") + labs(title = "Violin: distribution shape")
p4 <- ggplot(df, aes(study_hours, exam_score, colour = method)) + geom_point(alpha = .6) + geom_smooth(aes(group = 1), method = "lm", colour = "black") +
  labs(title = "Scatter: study hours vs score")
pr <- aggregate(passed ~ program, df, function(v) mean(v) * 100)
p5 <- ggplot(pr, aes(program, passed)) + geom_col(fill = "orange") + labs(title = "Bar: pass rate by programme (%)", y = "% passed")
p6 <- ggplot(sales, aes(date, sales)) + geom_line(colour = "steelblue") + labs(title = "Line: monthly sales")
for (nm in c("hist", "box", "violin", "scatter", "bar", "line")) {
  p <- get(paste0("p", match(nm, c("hist", "box", "violin", "scatter", "bar", "line"))))
  save_plot(nm, print(p), width = 900, height = 600)
  cat("Saved r_", nm, ".png\n", sep = "")
}
save_plot("lattice_bwplot", print(lattice::bwplot(exam_score ~ method, data = df, main = "lattice boxplot")))
save_plot("scatter_matrix", pairs(df[, c("study_hours", "attendance_pct", "prior_gpa", "exam_score")], pch = 16, col = rgb(0, 0, 1, .3)), width = 900, height = 900)

means <- aggregate(exam_score ~ method, df, mean)
bad  <- ggplot(means, aes(method, exam_score)) + geom_col(fill = "grey60") + coord_cartesian(ylim = c(55, 67)) + labs(title = "Misleading: truncated axis")
good <- ggplot(means, aes(method, exam_score)) + geom_col(fill = "steelblue") + labs(title = "Honest: axis starts at 0")
save_plot("misleading_axis", print(bad)); save_plot("honest_axis", print(good))

if (opt("plotly") && opt("htmlwidgets")) {
  w <- tryCatch(plotly::ggplotly(p4), error = function(e) NULL)
  if (!is.null(w)) {
    htmlwidgets::saveWidget(w, file.path(normalizePath(OUT), "r_interactive_scatter.html"), selfcontained = FALSE)
    cat("Saved r_interactive_scatter.html (open in a browser)\n")
  }
}

h("INTERPRETATION")
cat("- The histogram shows an approximately bell-shaped score distribution.\n")
cat("- Boxplot/violin show method C shifts the whole distribution upward.\n")
cat("- The scatter plot shows a positive linear relationship between study hours and score.\n")
cat("- Truncating a bar chart's axis exaggerates differences: always start bars at zero.\n")
end_topic()
