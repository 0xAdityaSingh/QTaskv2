using System;
using System.Windows.Forms;

namespace Raytracer
{
    /// <summary>
    /// Main form for the application.
    /// </summary>
    public partial class MainForm : Form
    {
        private readonly NewsService _newsService;

        /// <summary>
        /// Initializes a new instance of the <see cref="MainForm"/> class.
        /// </summary>
        /// <param name="newsService">The news service dependency.</param>
        public MainForm(NewsService newsService)
        {
            InitializeComponent();
            _newsService = newsService ?? throw new ArgumentNullException(nameof(newsService));
        }

        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this._isParallelCheckBox = new System.Windows.Forms.CheckBox();
            this._startButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // _isParallelCheckBox
            // 
            this._isParallelCheckBox.AutoSize = true;
            this._isParallelCheckBox.Location = new System.Drawing.Point(12, 12);
            this._isParallelCheckBox.Name = "_isParallelCheckBox";
            this._isParallelCheckBox.Size = new System.Drawing.Size(86, 17);
            this._isParallelCheckBox.TabIndex = 0;
            this._isParallelCheckBox.Text = "Use Parallelism";
            this._isParallelCheckBox.UseVisualStyleBackColor = true;
            // 
            // _startButton
            // 
            this._startButton.Location = new System.Drawing.Point(12, 35);
            this._startButton.Name = "_startButton";
            this._startButton.Size = new System.Drawing.Size(75, 23);
            this._startButton.TabIndex = 1;
            this._startButton.Text = "Start";
            this._startButton.UseVisualStyleBackColor = true;
            this._startButton.Click += new System.EventHandler(this._startButton_Click);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 261);
            this.Controls.Add(this._startButton);
            this.Controls.Add(this._isParallelCheckBox);
            this.Name = "MainForm";
            this.Text = "Raytracer";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.CheckBox _isParallelCheckBox;
        private System.Windows.Forms.Button _startButton;

        /// <summary>
        /// Handles the click event for the start button.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The event arguments.</param>
        private void _startButton_Click(object sender, EventArgs e)
        {
            try
            {
                if (_isParallelCheckBox.Checked)
                {
                    // Use parallel processing
                    _newsService.ProcessNewsInParallel();
                }
                else
                {
                    // Use sequential processing
                    _newsService.ProcessNewsSequentially();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"An error occurred: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}