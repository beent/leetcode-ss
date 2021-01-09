/// <summary>
/// 1476. 子矩形查询
/// </summary>
public class SubrectangleQueries
{
    private int[][] rect;
    private int[][] operateHistory = new int[501][];
    private int historyIndex = 0;
    public SubrectangleQueries(int[][] rectangle)
    {
        this.rect = rectangle;
    }

    public void UpdateSubrectangle(int row1, int col1, int row2, int col2, int newValue)
    {
        operateHistory[historyIndex++] = new int[] { row1, col1, row2, col2, newValue };
    }

    public int GetValue(int row, int col)
    {
        int val = rect[row][col];
        for (int i = 0; i < historyIndex; i++)
        {
            int[] ints = operateHistory[i];
            if (row >= ints[0] && row <= ints[2] && col >= ints[1] && col <= ints[3])
            {
                val = ints[4];
            }
        }
        return val;
    }
}

