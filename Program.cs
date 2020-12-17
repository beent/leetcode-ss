using System;

namespace Leetcode
{
    class Program
    {
        static void Main(string[] args)
        {
            Source source = new Source();
            int[] nums = new int[] { 1, 2, 3, 4 };
            int target = 3;
            int N = 10;
            string s = "32";
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
            string[] strs = new string[] { "eat", "tea", "tan", "ate", "nat", "bat" };

            int[][] cuboids = new int[][] { new int[] { 50, 45, 20 }, new int[] { 95, 37, 53 }, new int[] { 45, 23, 12 } };

            int[][] grid = new int[][] { new int[] { 1, 0 }, new int[] { 1, 1 } };
            // source.Rotate(nums, target);
            var result = source.MonotoneIncreasingDigits(N);
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
