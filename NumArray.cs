namespace Leetcode
{
    public class NumArray
    {
        // 线段树实现
        int[] tree;
        int n;

        private void BuildTree(int[] nums)
        {
            for (int i = n, j = 0; i < 2 * n; i++, j++)
            {
                tree[i] = nums[j];
            }
            for (int i = n - 1; i > 0; i--)
            {
                tree[i] = tree[i * 2] + tree[i * 2 + 1];
            }
        }

        public NumArray(int[] nums)
        {
            if (nums.Length > 0)
            {
                n = nums.Length;
                tree = new int[n * 2];
                BuildTree(nums);
            }
        }

        public void Update(int i, int val)
        {
            i += n;
            tree[i] = val;
            while (i > 0)
            {
                int l = i, r = i;
                if (i % 2 == 0)
                {
                    r = i + 1;
                }
                else
                {
                    l = i - 1;
                }
                tree[i / 2] = tree[l] + tree[r];
                i /= 2;
            }
        }

        public int SumRange(int i, int j)
        {
            i += n;
            j += n;
            int sum = 0;
            while (i <= j)
            {
                if ((i % 2) == 1)
                {
                    sum += tree[i++];
                }
                if ((j % 2) == 0)
                {
                    sum += tree[j--];
                }
                i /= 2;
                j /= 2;
            }
            return sum;
        }

        // 操作原数组
        // int[] arr;
        // public NumArray(int[] nums)
        // {
        //     arr = nums;
        // }

        // public void Update(int i, int val)
        // {
        //     arr[i] = val;
        // }

        // public int SumRange(int i, int j)
        // {
        //     int res = 0;
        //     for (; i <= j; i++)
        //     {
        //         res += arr[i];
        //     }
        //     return res;
        // }
    }
}