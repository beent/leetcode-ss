using System;

namespace Leetcode
{
    class Program
    {
        static void Main(string[] args)
        {
            Source source = new Source();
            int[] nums = new int[] { 0, 1, 2, 4, 5, 6, 8, 10, 12 };
            int target = 3;
            string s = "a good   example";
            char[][] board = new char[][] { new char[]{'5', '3', '.', '.', '7', '.', '.', '.', '.'},
                                            new char[]{'6', '.', '.', '1', '9', '5', '.', '.', '.'},
                                            new char[]{'.', '9', '8', '.', '.', '.', '.', '6', '.'},
                                            new char[]{'8', '.', '.', '.', '6', '.', '.', '.', '3'},
                                            new char[]{'4', '.', '.', '8', '.', '3', '.', '.', '1'},
                                            new char[]{'7', '.', '.', '.', '2', '.', '.', '.', '6'},
                                            new char[]{'.', '6', '.', '.', '.', '.', '2', '8', '.'},
                                            new char[]{'.', '.', '.', '4', '1', '9', '.', '.', '5'},
                                            new char[]{'.', '.', '.', '.', '8', '.', '.', '7', '9'}
                                            };

            int[][] grid = new int[][] { new int[] { 1, 0 }, new int[] { 1, 1 } };
            // source.Rotate(nums, target);
            var result = source.CountServers(grid);
            Console.WriteLine($"Result: {result}");
            // NumArray arr = new NumArray(nums);

            // arr.Update(0, 3);
            // Console.WriteLine(arr.SumRange(1, 1));
            // Console.WriteLine(arr.SumRange(0, 1));
            // arr.Update(1, -3);
            // Console.WriteLine(arr.SumRange(0, 1));
            Console.WriteLine("End!");

        }
    }
}
